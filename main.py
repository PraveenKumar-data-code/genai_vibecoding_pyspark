"""Generate synthetic e-commerce data as Parquet files."""

import logging
from time import perf_counter

import src.config as config
from src.data_generator import SyntheticDataGenerator

LOGGER: logging.Logger = logging.getLogger(__name__)


def main() -> int:
    """Generate the configured datasets and write them to the raw data directory."""
    config.configure_logging()
    try:
        generator: SyntheticDataGenerator = SyntheticDataGenerator(
            num_customers=config.DEFAULT_CUSTOMER_COUNT,
            num_products=config.DEFAULT_PRODUCT_COUNT,
            num_orders=config.DEFAULT_ORDER_COUNT,
            seed=config.DEFAULT_SEED,
        )
        generation_started: float = perf_counter()
        customers, products, orders = generator.generate_all()
        generation_seconds: float = perf_counter() - generation_started

        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        datasets = {
            "customers": customers,
            "products": products,
            "orders": orders,
        }
        output_paths = {
            name: config.RAW_DATA_DIR / f"{name}.parquet"
            for name in datasets
        }
        for name, dataframe in datasets.items():
            dataframe.to_parquet(output_paths[name], index=False)

        print(f"Data generation time: {generation_seconds:.2f} seconds")
        for name, path in output_paths.items():
            print(f"{path.name}: {path.stat().st_size:,} bytes")
    except Exception:
        LOGGER.exception("Failed to generate or save synthetic data")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
