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
        pl.concat_list([pl.col('Incremental Paid Loss: Loss Ratio').shift_and_fill(i, _MASK_VALUE) for i in range(1, 9 + 1)][::-1])
            .alias('Incremental Loss: Input')
    ).over(['Group Code', 'Accident Year', 'Line of Business']),

    (
        pl.concat_list([pl.col('Case Reserves: Loss Ratio').shift_and_fill(i, _MASK_VALUE) for i in range(1, 9 + 1)][::-1])
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

training_data = model_data[('Train', 'Private passenger auto')].drop(['Fitting Bucket', 'Line of Business']).to_numpy()
validation_data = model_data[('Validation', 'Private passenger auto')].drop(['Fitting Bucket', 'Line of Business']).to_numpy()
testing_data = model_data['Test'].drop('Fitting Bucket').to_numpy()