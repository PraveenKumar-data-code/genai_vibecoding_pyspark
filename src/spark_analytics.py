"""Analyze generated e-commerce CSV datasets with PySpark."""

import argparse
import logging
from pathlib import Path
from typing import Sequence

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.window import WindowSpec
from pyspark.sql.types import (
    DateType,
    DecimalType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SPARK_APP_NAME,
    configure_logging,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
ORDERS_SCHEMA: StructType = StructType(
    [
        StructField("order_id", StringType(), nullable=False),
        StructField("customer_id", StringType(), nullable=False),
        StructField("product_id", StringType(), nullable=False),
        StructField("quantity", IntegerType(), nullable=False),
        StructField("unit_price", DecimalType(10, 2), nullable=False),
        StructField("order_date", DateType(), nullable=False),
        StructField("status", StringType(), nullable=False),
    ]
)
CUSTOMERS_SCHEMA: StructType = StructType(
    [
        StructField("customer_id", StringType(), nullable=False),
        StructField("first_name", StringType(), nullable=False),
        StructField("last_name", StringType(), nullable=False),
        StructField("email", StringType(), nullable=False),
        StructField("city", StringType(), nullable=False),
        StructField("state", StringType(), nullable=False),
        StructField("signup_date", DateType(), nullable=False),
    ]
)
PRODUCTS_SCHEMA: StructType = StructType(
    [
        StructField("product_id", StringType(), nullable=False),
        StructField("product_name", StringType(), nullable=False),
        StructField("category", StringType(), nullable=False),
        StructField("unit_price", DecimalType(10, 2), nullable=False),
    ]
)


def build_spark_session(app_name: str = SPARK_APP_NAME) -> SparkSession:
    """Create a local Spark session for running the analytics pipeline."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


class SalesAnalytics:
    """Run sales analytics over order and product Spark DataFrames."""

    @staticmethod
    def create_spark_session() -> SparkSession:
        """Create a configured local Spark session for sales analytics."""
        return build_spark_session()

    @staticmethod
    def load_parquet(path: str | Path) -> DataFrame:
        """Load a Parquet file or directory into a Spark DataFrame."""
        spark: SparkSession = build_spark_session()
        return spark.read.parquet(str(path))

    @staticmethod
    def _join_sales_data(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join orders to products and add revenue using the available price column."""
        if "quantity" not in orders_df.columns or "product_id" not in orders_df.columns:
            raise ValueError("orders_df must contain quantity and product_id columns")
        if "product_id" not in products_df.columns:
            raise ValueError("products_df must contain a product_id column")

        if "unit_price" in orders_df.columns:
            price_column = F.col("orders.unit_price")
        else:
            product_price_column: str | None = next(
                (
                    column
                    for column in ("price", "unit_price")
                    if column in products_df.columns
                ),
                None,
            )
            if product_price_column is None:
                raise ValueError(
                    "orders_df or products_df must contain a price column "
                    "('unit_price' or 'price')"
                )
            price_column = F.col(f"products.{product_price_column}")

        return (
            orders_df.alias("orders")
            .join(
                products_df.alias("products"),
                F.col("orders.product_id") == F.col("products.product_id"),
                "inner",
            )
            .withColumn(
                "revenue",
                F.col("orders.quantity").cast("double") * price_column.cast("double"),
            )
        )

    @staticmethod
    def top_customers_by_revenue(
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the top N customers ranked by total order revenue."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if "customer_id" not in orders_df.columns:
            raise ValueError("orders_df must contain a customer_id column")

        sales: DataFrame = SalesAnalytics._join_sales_data(orders_df, products_df)
        return (
            sales.groupBy(F.col("orders.customer_id").alias("customer_id"))
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )

    @staticmethod
    def sales_by_category(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return revenue and units sold grouped by product category."""
        if "category" not in products_df.columns:
            raise ValueError("products_df must contain a category column")

        sales: DataFrame = SalesAnalytics._join_sales_data(orders_df, products_df)
        return (
            sales.groupBy(F.col("products.category").alias("category"))
            .agg(
                F.round(F.sum("revenue"), 2).alias("revenue"),
                F.sum(F.col("orders.quantity")).alias("units_sold"),
            )
            .orderBy(F.desc("revenue"))
        )

    @staticmethod
    def monthly_trends(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return monthly revenue and month-over-month revenue growth percentage."""
        if "order_date" not in orders_df.columns:
            raise ValueError("orders_df must contain an order_date column")

        sales: DataFrame = SalesAnalytics._join_sales_data(orders_df, products_df)
        monthly_revenue: DataFrame = (
            sales.withColumn("month", F.date_format("orders.order_date", "yyyy-MM"))
            .groupBy("month")
            .agg(F.round(F.sum("revenue"), 2).alias("revenue"))
        )
        monthly_window: WindowSpec = Window.orderBy("month")
        trends: DataFrame = monthly_revenue.withColumn(
            "previous_month_revenue",
            F.lag("revenue").over(monthly_window),
        )
        return (
            trends.withColumn(
                "month_over_month_growth_pct",
                F.when(
                    F.col("previous_month_revenue").isNull()
                    | (F.col("previous_month_revenue") == 0),
                    F.lit(None).cast("double"),
                ).otherwise(
                    F.round(
                        (
                            (F.col("revenue") - F.col("previous_month_revenue"))
                            / F.col("previous_month_revenue")
                        )
                        * 100,
                        2,
                    )
                ),
            )
            .drop("previous_month_revenue")
            .orderBy("month")
        )


def _read_csv(spark: SparkSession, path: Path, schema: StructType) -> DataFrame:
    """Read a headered CSV file using its explicit Spark schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    return (
        spark.read.option("header", True)
        .option("dateFormat", "yyyy-MM-dd")
        .schema(schema)
        .csv(str(path))
    )


def analyze_orders(
    spark: SparkSession,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    """Calculate and save monthly sales, product performance, and customer spend."""
    orders: DataFrame = _read_csv(spark, raw_dir / "orders.csv", ORDERS_SCHEMA)
    customers: DataFrame = _read_csv(
        spark, raw_dir / "customers.csv", CUSTOMERS_SCHEMA
    )
    products: DataFrame = _read_csv(spark, raw_dir / "products.csv", PRODUCTS_SCHEMA)
    revenue_orders: DataFrame = orders.filter(
        F.col("status").isin("completed", "shipped")
    ).withColumn("revenue", F.col("quantity") * F.col("unit_price"))

    monthly_sales: DataFrame = (
        revenue_orders.withColumn("month", F.date_format("order_date", "yyyy-MM"))
        .groupBy("month")
        .agg(
            F.count("order_id").alias("order_count"),
            F.round(F.sum("revenue"), 2).alias("revenue"),
        )
        .orderBy("month")
    )
    top_products: DataFrame = (
        revenue_orders.join(products, on="product_id", how="inner")
        .groupBy("product_id", "product_name", "category")
        .agg(
            F.sum("quantity").alias("units_sold"),
            F.round(F.sum("revenue"), 2).alias("revenue"),
        )
        .orderBy(F.desc("revenue"))
    )
    customer_spend: DataFrame = (
        revenue_orders.join(customers, on="customer_id", how="inner")
        .groupBy("customer_id", "first_name", "last_name", "city", "state")
        .agg(
            F.count("order_id").alias("order_count"),
            F.round(F.sum("revenue"), 2).alias("total_spend"),
        )
        .orderBy(F.desc("total_spend"))
    )
    results: dict[str, DataFrame] = {
        "monthly_sales": monthly_sales,
        "top_products": top_products,
        "customer_spend": customer_spend,
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for name, dataframe in results.items():
        output_path: Path = processed_dir / name
        dataframe.write.mode("overwrite").option("header", True).csv(str(output_path))
        output_paths[name] = output_path
        LOGGER.info("Wrote %s insights to %s", name, output_path)
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for Spark analytics."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Analyze generated e-commerce CSV datasets with PySpark."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Spark analytics and stop the Spark session when finished."""
    configure_logging()
    args: argparse.Namespace = parse_args(argv)
    spark: SparkSession = build_spark_session()
    try:
        analyze_orders(spark, raw_dir=args.raw_dir, processed_dir=args.processed_dir)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())