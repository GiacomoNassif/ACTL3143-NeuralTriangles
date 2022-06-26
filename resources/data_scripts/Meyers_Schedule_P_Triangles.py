"""
Data sourced from https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p
"""
import polars as pl
from functools import reduce
import os

"""
DATA_INFO is of the form: {DataName: DataURL}
"""
_DATA_INFO: dict[str, str] = {
    'Private passenger auto': 'https://www.casact.org/sites/default/files/2021-04/ppauto_pos.csv',
    "Workers' compensation": 'https://www.casact.org/sites/default/files/2021-04/wkcomp_pos.csv',
    'Commercial Auto': 'https://www.casact.org/sites/default/files/2021-04/comauto_pos.csv',
    'Medical Malpractice': 'https://www.casact.org/sites/default/files/2021-04/medmal_pos.csv',
    'Product Liability': 'https://www.casact.org/sites/default/files/2021-04/prodliab_pos.csv',
    'Other Liability': 'https://www.casact.org/sites/default/files/2021-04/othliab_pos.csv'
}

_COLUMN_NAMES = (
    'Group Code',
    'Group Name',
    'Accident Year',
    'Development Year',
    'Development Lag',
    'Incurred Loss',
    'Cumulative Paid Loss',
    'Bulk Loss',
    'Earned Premium Direct',
    'Earned Premium Ceded',
    'Earned Premium Net',
    'Single',
    'Posted Reserve 1997'
)

_SAVED_FILE_NAME = 'SchedulePTriangles.pq'

def extract_data(data_url: str) -> pl.DataFrame:
    return pl.read_csv(data_url)


def rename_columns(data: pl.DataFrame) -> pl.DataFrame:
    return data.rename(dict(zip(data.columns, _COLUMN_NAMES)))


def transform_data(data: pl.DataFrame, data_name: str) -> pl.DataFrame:
    data = rename_columns(data)

    return data.with_column(
        pl.lit(data_name).alias('Line of Business')
    )


def extract_transform(data_name: str, data_url: str) -> pl.DataFrame:
    data = extract_data(data_url)
    return transform_data(data, data_name)


def ETL() -> pl.DataFrame:
    # Load in all the data (lazily because its a map)
    extracted_data = map(lambda info: extract_transform(*info), _DATA_INFO.items())

    # Stack the data on top of each other
    stacked_data = reduce(lambda df1, df2: df1.vstack(df2), extracted_data)

    # Add the calender year column
    stacked_data = stacked_data.with_column(
        (pl.col('Accident Year') + pl.col('Development Lag') - 1).alias('Calender Year')
    )

    # This is just cumulative paid at time t - cumulative paid at time t-1.
    stacked_data = stacked_data.sort('Development Lag').with_column(
        (
                pl.col('Cumulative Paid Loss') -
                pl.col('Cumulative Paid Loss').shift_and_fill(periods=1, fill_value=0)
        ).over(['Group Code', 'Accident Year', 'Line of Business']).alias('Incremental Paid Loss')
    )

    # Down cast some string columns for efficiency
    stacked_data = stacked_data.with_columns([
        pl.col('Group Name').cast(pl.Categorical),
        pl.col('Line of Business').cast(pl.Categorical)
    ])

    return stacked_data


def get_schedule_P_triangle_data(cache: bool = True) -> pl.LazyFrame:
    if _SAVED_FILE_NAME in os.listdir('resources/data'):
        return pl.scan_parquet(f'resources/data/{_SAVED_FILE_NAME}')

    data = ETL()

    if cache:
        data.write_parquet('./resources/data/SchedulePTriangles.pq')

    return data.lazy()

