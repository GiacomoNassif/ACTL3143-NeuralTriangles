"""
Add features to the Meyers-P dataset
"""
import polars as pl
from resources.data_scripts.Meyers_Schedule_P_Triangles import get_schedule_P_triangle_data
import os

_FILE_PATH = 'resources/data/Decorated_P_Triangles.pq'


def get_decorated_p_triangles(cache: bool = True) -> pl.LazyFrame:
    if os.path.exists(_FILE_PATH):
        return pl.scan_parquet(_FILE_PATH)

    triangle_df = get_schedule_P_triangle_data(cache)

    decorated_df = triangle_df.with_column(
        (pl.col('Incurred Loss') - pl.col('Cumulative Paid Loss')).alias('Case Reserves')
    )

    # Make loss ratio columns
    loss_ratio_cols = [
        (pl.col(col_name) / pl.col('Earned Premium Net')).alias(f'{col_name}: Loss Ratio')
        for col_name in ('Incremental Paid Loss', 'Cumulative Paid Loss', 'Case Reserves')
    ]

    decorated_df = decorated_df.with_columns(loss_ratio_cols)

    # Split out train and test data
    decorated_df = decorated_df.with_column(
        pl.when(pl.col('Development Lag') == 1)
            .then('Discard')
            .when(pl.col('Calender Year') <= 1995)
            .then('Train')
            .when(pl.col('Calender Year') <= 1997)
            .then('Validation')
            .otherwise('Test')
            .alias('Fitting Bucket')
            .cast(pl.Categorical)
    )

    if cache:
        decorated_df.collect().write_parquet(_FILE_PATH)

    return decorated_df
