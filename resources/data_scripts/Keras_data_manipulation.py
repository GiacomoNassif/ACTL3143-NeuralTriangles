import numpy as np
import polars as pl
import pandas as pd
from resources.data_scripts.Add_Features_To_Meyers_P import get_decorated_p_triangles

USED_COMPANY_URL = r'https://www.casact.org/sites/default/files/2021-03/Monograph_Tables_and_Scripts.xlsx'

_MASK_VALUE = -99

p_triangle_df = get_decorated_p_triangles()

COMPANY_MAPPINGS = {
    'CA': 'Commercial Auto',
    'PA': 'Private passenger auto',
    'WC': "Workers' compensation",
    'OL': 'Other Liability'
}

used_companies_raw = pd.read_excel(
    USED_COMPANY_URL,
    sheet_name='Multi Mack Paid',
    skiprows=4,
    usecols='A:B'
)

used_companies = pl.DataFrame([
    pl.Series('Line of Business', used_companies_raw['Line'].map(COMPANY_MAPPINGS)),
    pl.Series('Group Code', used_companies_raw['Group Code']),
]).lazy()

keras_data = p_triangle_df.sort('Development Lag').with_columns([
    (
        pl.concat_list(
            [pl.col('Incremental Paid Loss: Loss Ratio').shift_and_fill(i, _MASK_VALUE) for i in range(1, 9 + 1)][::-1])
            .alias('Incremental Loss: Input')
    ).over(['Group Code', 'Accident Year', 'Line of Business']),

    (
        pl.concat_list(
            [pl.col('Case Reserves: Loss Ratio').shift_and_fill(i, _MASK_VALUE) for i in range(1, 9 + 1)][::-1])
            .alias('Case Reserves: Input')
    ).over(['Group Code', 'Accident Year', 'Line of Business']),

    (
        pl.concat_list([pl.col('Incremental Paid Loss: Loss Ratio').shift_and_fill(-i, _MASK_VALUE) for i in range(9)])
            .alias('Incremental Loss: Output')
    ).over(['Group Code', 'Accident Year', 'Line of Business']),

    (
        pl.concat_list([pl.col('Case Reserves: Loss Ratio').shift_and_fill(-i, _MASK_VALUE) for i in range(9)])
            .alias('Case Reserves: Output')
    ).over(['Group Code', 'Accident Year', 'Line of Business']),

])

pl.toggle_string_cache(True)
used_companies = used_companies.with_column(pl.col('Line of Business').cast(pl.Categorical))
keras_data = keras_data.with_column(pl.col('Line of Business').cast(pl.Categorical))

keras_data = used_companies.join(keras_data, on=['Line of Business', 'Group Code'])

keras_data = keras_data.select([
    'Line of Business',
    'Fitting Bucket',
    'Group Code',
    'Incremental Loss: Input',
    'Case Reserves: Input',
    'Incremental Loss: Output',
    'Case Reserves: Output'
])

model_data = keras_data.collect().partition_by(['Fitting Bucket', 'Line of Business'], as_dict=True)


def extract_used_columns(df: pl.DataFrame) -> np.ndarray:
    return df.drop(['Fitting Bucket', 'Line of Business']).to_numpy()


def map_numpy_to_keras_data(data: np.ndarray):
    company_codes: np.ndarray = data[:, 0].reshape(-1, 1).astype(int)
    loss_paid: np.ndarray = np.vstack(data[:, 1]).reshape((-1, 9, 1))
    case_reserves: np.ndarray = np.vstack(data[:, 2]).reshape((-1, 9, 1))

    return company_codes, loss_paid, case_reserves


def extract_keras_data(df: pl.DataFrame):
    numpy_data = extract_used_columns(df)
    return map_numpy_to_keras_data(numpy_data)


def get_insurance_line_data(insurance_line: str):
    # Kevin Kuo combines the train and validation set to be used in training. Then also has early stopping on the
    # validation set. Very odd but we will do the same.
    training_data = model_data[('Train', insurance_line)].vstack(model_data[('Validation', insurance_line)])
    training_data = extract_keras_data(training_data)

    validation_data = extract_keras_data(model_data[('Validation', insurance_line)])

    return training_data, validation_data
