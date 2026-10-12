"""Run the Parquet-based sales analytics pipeline."""

from collections.abc import Callable
from time import perf_counter

from pyspark.sql import DataFrame, SparkSession

from src.config import RAW_DATA_DIR
from src.spark_analytics import SalesAnalytics


def display_timed_result(
    operation_name: str,
    operation: Callable[[], DataFrame],
) -> None:
    """Run an analytics operation, display its result, and print elapsed time."""
    started_at: float = perf_counter()
    result: DataFrame = operation()
    result.show()
    elapsed_seconds: float = perf_counter() - started_at
    print(f"{operation_name} execution time: {elapsed_seconds:.2f} seconds")


def main() -> None:
    """Load the raw Parquet datasets and display all sales analytics."""
    analytics: SalesAnalytics = SalesAnalytics()
    spark: SparkSession = analytics.create_spark_session()
    try:
        customers_df: DataFrame = analytics.load_parquet(
            RAW_DATA_DIR / "customers.parquet"
        )
        products_df: DataFrame = analytics.load_parquet(
            RAW_DATA_DIR / "products.parquet"
        )
        orders_df: DataFrame = analytics.load_parquet(RAW_DATA_DIR / "orders.parquet")
        if "customer_id" not in customers_df.columns:
            raise ValueError("customers.parquet must contain a customer_id column")

        display_timed_result(
            "Top customers by revenue",
            lambda: analytics.top_customers_by_revenue(orders_df, products_df),
        )
        display_timed_result(
            "Sales by category",
            lambda: analytics.sales_by_category(orders_df, products_df),
        )
        display_timed_result(
            "Monthly trends",
            lambda: analytics.monthly_trends(orders_df, products_df),
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
