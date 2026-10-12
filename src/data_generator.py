"""Generate reproducible fake e-commerce data as CSV files."""

import argparse
import csv
import logging
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from src.config import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_ORDER_COUNT,
    DEFAULT_PRODUCT_COUNT,
    DEFAULT_SEED,
    RAW_DATA_DIR,
    configure_logging,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
CUSTOMER_FIELDS: tuple[str, ...] = (
    "customer_id",
    "first_name",
    "last_name",
    "email",
    "city",
    "state",
    "signup_date",
)
PRODUCT_FIELDS: tuple[str, ...] = (
    "product_id",
    "product_name",
    "category",
    "unit_price",
)
ORDER_FIELDS: tuple[str, ...] = (
    "order_id",
    "customer_id",
    "product_id",
    "quantity",
    "unit_price",
    "order_date",
    "status",
)
PRODUCT_CATEGORIES: tuple[str, ...] = (
    "Electronics",
    "Home",
    "Books",
    "Clothing",
    "Sports",
    "Beauty",
)
ORDER_STATUSES: tuple[str, ...] = (
    "completed",
    "shipped",
    "pending",
    "cancelled",
)
REFERENCE_DATE: date = date(2025, 12, 31)


class SyntheticDataGenerator:
    """Generate synthetic e-commerce customers, products, and orders as DataFrames."""

    PRODUCT_CATEGORIES: tuple[str, ...] = (
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Books",
    )

    def __init__(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        seed: int | None = DEFAULT_SEED,
        pareto_shape: float = 1.16,
        reference_date: date | None = None,
    ) -> None:
        """Initialize dataset sizes, seeded random generators, and the date range."""
        if min(num_customers, num_products, num_orders) < 0:
            raise ValueError("dataset sizes must be non-negative")
        if pareto_shape <= 0:
            raise ValueError("pareto_shape must be greater than zero")

        self.num_customers: int = num_customers
        self.num_products: int = num_products
        self.num_orders: int = num_orders
        self.pareto_shape: float = pareto_shape
        self.reference_date: date = reference_date or date.today()
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.fake: Faker = Faker("en_US")
        if seed is not None:
            self.fake.seed_instance(seed)
        self._customers: pd.DataFrame | None = None
        self._products: pd.DataFrame | None = None
        self._high_activity_customer_ids: set[str] = set()

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer details with normally distributed ages around 35."""
        LOGGER.info("Generating %d synthetic customers", self.num_customers)
        customer_rows: list[dict[str, object]] = []
        earliest_registration: date = self.reference_date - timedelta(days=365 * 5)
        for _ in tqdm(
            range(self.num_customers), desc="Generating customers", unit="customer"
        ):
            customer_rows.append(
                {
                    "name": self.fake.name(),
                    "email": self.fake.unique.email(),
                    "city": self.fake.city(),
                    "country": self.fake.country(),
                }
            )

        registration_offsets: np.ndarray = self.rng.integers(
            0, 365 * 5 + 1, size=self.num_customers
        )
        customer_ages: np.ndarray = np.clip(
            np.rint(self.rng.normal(loc=35, scale=10, size=self.num_customers)),
            18,
            90,
        ).astype(np.int16)
        registration_dates: pd.DatetimeIndex = pd.to_datetime(
            pd.Timestamp(earliest_registration)
            + pd.to_timedelta(365 * 5 - registration_offsets, unit="D")
        )
        customers: pd.DataFrame = pd.DataFrame(
            customer_rows, columns=["name", "email", "city", "country"]
        )
        customers.insert(
            0,
            "customer_id",
            [f"C{index:07d}" for index in range(1, self.num_customers + 1)],
        )
        customers["age"] = customer_ages
        customers["registration_date"] = registration_dates
        self._customers = customers[
            [
                "customer_id",
                "name",
                "email",
                "age",
                "city",
                "country",
                "registration_date",
            ]
        ]
        LOGGER.info("Generated customer DataFrame with %d rows", len(self._customers))
        return self._customers

    def generate_products(self) -> pd.DataFrame:
        """Generate products with prices, stock counts, categories, and ratings."""
        LOGGER.info("Generating %d synthetic products", self.num_products)
        product_names: list[str] = [
            self.fake.catch_phrase()
            for _ in tqdm(
                range(self.num_products), desc="Generating products", unit="product"
            )
        ]
        products: pd.DataFrame = pd.DataFrame(
            {
                "product_id": [
                    f"P{index:06d}" for index in range(1, self.num_products + 1)
                ],
                "name": product_names,
                "category": self.rng.choice(
                    self.PRODUCT_CATEGORIES, size=self.num_products
                ),
                "price": np.round(
                    self.rng.uniform(10.0, 500.0, size=self.num_products), 2
                ),
                "stock": self.rng.integers(0, 1_001, size=self.num_products),
                "rating": self.rng.integers(1, 6, size=self.num_products),
            }
        )
        self._products = products
        LOGGER.info("Generated product DataFrame with %d rows", len(products))
        return products

    def generate_orders(
        self,
        customers: pd.DataFrame | None = None,
        products: pd.DataFrame | None = None,
        chunk_size: int = 100_000,
    ) -> pd.DataFrame:
        """Generate orders, allocating 80% to the top 20% of Pareto-ranked customers."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.num_orders == 0:
            self._high_activity_customer_ids = set()
            return pd.DataFrame(
                columns=[
                    "order_id",
                    "customer_id",
                    "product_id",
                    "quantity",
                    "order_date",
                ]
            )

        customer_frame: pd.DataFrame = customers if customers is not None else (
            self._customers if self._customers is not None else self.generate_customers()
        )
        product_frame: pd.DataFrame = products if products is not None else (
            self._products if self._products is not None else self.generate_products()
        )
        if customer_frame.empty or product_frame.empty:
            raise ValueError("customers and products must not be empty when generating orders")
        if "customer_id" not in customer_frame or "product_id" not in product_frame:
            raise ValueError("customer and product DataFrames must include their ID columns")

        customer_ids: np.ndarray = customer_frame["customer_id"].to_numpy()
        product_ids: np.ndarray = product_frame["product_id"].to_numpy()
        customer_count: int = len(customer_ids)
        high_activity_count: int = max(1, math.ceil(customer_count * 0.2))
        pareto_weights: np.ndarray = self.rng.pareto(
            self.pareto_shape, size=customer_count
        ) + 1.0
        high_activity_indices: np.ndarray = np.argpartition(
            pareto_weights, customer_count - high_activity_count
        )[-high_activity_count:]
        regular_indices: np.ndarray = np.setdiff1d(
            np.arange(customer_count), high_activity_indices, assume_unique=True
        )
        self._high_activity_customer_ids = set(
            customer_ids[high_activity_indices].tolist()
        )

        high_activity_order_count: int = int(round(self.num_orders * 0.8))
        if regular_indices.size == 0:
            high_activity_order_count = self.num_orders
        regular_order_count: int = self.num_orders - high_activity_order_count
        assignment_chunks: list[np.ndarray] = []
        with tqdm(total=self.num_orders, desc="Assigning orders", unit="order") as bar:
            for cohort_indices, cohort_count in (
                (high_activity_indices, high_activity_order_count),
                (regular_indices, regular_order_count),
            ):
                if cohort_count == 0:
                    continue
                cohort_weights: np.ndarray = pareto_weights[cohort_indices]
                cohort_probabilities: np.ndarray = cohort_weights / cohort_weights.sum()
                for start in range(0, cohort_count, chunk_size):
                    size: int = min(chunk_size, cohort_count - start)
                    assignment_chunks.append(
                        self.rng.choice(
                            cohort_indices,
                            size=size,
                            replace=True,
                            p=cohort_probabilities,
                        )
                    )
                    bar.update(size)

        customer_indices: np.ndarray = np.concatenate(assignment_chunks)
        self.rng.shuffle(customer_indices)
        product_indices: np.ndarray = self.rng.integers(
            0, len(product_ids), size=self.num_orders
        )
        quantities: np.ndarray = self.rng.integers(1, 11, size=self.num_orders)
        order_offsets: np.ndarray = self.rng.integers(
            0, 366, size=self.num_orders
        )
        order_dates: pd.DatetimeIndex = pd.to_datetime(
            pd.Timestamp(self.reference_date)
            - pd.to_timedelta(order_offsets, unit="D")
        )
        order_ids: list[str] = [
            f"O{index:09d}"
            for index in tqdm(
                range(1, self.num_orders + 1), desc="Creating order IDs", unit="order"
            )
        ]
        orders: pd.DataFrame = pd.DataFrame(
            {
                "order_id": order_ids,
                "customer_id": customer_ids[customer_indices],
                "product_id": product_ids[product_indices],
                "quantity": quantities,
                "order_date": order_dates,
            }
        )
        LOGGER.info("Generated order DataFrame with %d rows", len(orders))
        return orders

    def generate_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return customer, product, and order DataFrames in that order."""
        customers: pd.DataFrame = self.generate_customers()
        products: pd.DataFrame = self.generate_products()
        orders: pd.DataFrame = self.generate_orders(customers, products)
        return customers, products, orders


def generate_customers(count: int, seed: int = DEFAULT_SEED) -> list[dict[str, str]]:
    """Create fictional customers with stable IDs and seeded fake details."""
    if count < 0:
        raise ValueError("count must be non-negative")

    fake: Faker = Faker("en_US")
    fake.seed_instance(seed)
    earliest_signup: date = REFERENCE_DATE - timedelta(days=365 * 5)
    customers: list[dict[str, str]] = []
    for index in range(1, count + 1):
        customers.append(
            {
                "customer_id": f"C{index:06d}",
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "email": fake.unique.email(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "signup_date": fake.date_between(
                    start_date=earliest_signup, end_date=REFERENCE_DATE
                ).isoformat(),
            }
        )
    return customers


def generate_products(count: int, seed: int = DEFAULT_SEED) -> list[dict[str, str]]:
    """Create fake products with stable IDs, categories, and prices."""
    if count < 0:
        raise ValueError("count must be non-negative")

    fake: Faker = Faker("en_US")
    fake.seed_instance(seed + 1)
    rng: random.Random = random.Random(seed + 1)
    return [
        {
            "product_id": f"P{index:05d}",
            "product_name": fake.catch_phrase(),
            "category": rng.choice(PRODUCT_CATEGORIES),
            "unit_price": f"{rng.uniform(5.0, 500.0):.2f}",
        }
        for index in range(1, count + 1)
    ]


def generate_orders(
    customer_ids: Sequence[str],
    products: Sequence[dict[str, str]],
    count: int,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, str]]:
    """Create fake orders linked to the supplied customers and products."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count and (not customer_ids or not products):
        raise ValueError("customers and products are required to generate orders")

    rng: random.Random = random.Random(seed + 2)
    earliest_order: date = REFERENCE_DATE - timedelta(days=365)
    orders: list[dict[str, str]] = []
    for index in range(1, count + 1):
        product: dict[str, str] = rng.choice(products)
        orders.append(
            {
                "order_id": f"O{index:07d}",
                "customer_id": rng.choice(customer_ids),
                "product_id": product["product_id"],
                "quantity": str(rng.randint(1, 5)),
                "unit_price": product["unit_price"],
                "order_date": (
                    earliest_order + timedelta(days=rng.randint(0, 365))
                ).isoformat(),
                "status": rng.choice(ORDER_STATUSES),
            }
        )
    return orders


def write_csv(
    path: Path,
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
) -> None:
    """Write rows to a UTF-8 CSV file with a header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer: csv.DictWriter[str] = csv.DictWriter(
            csv_file, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(
    output_dir: Path = RAW_DATA_DIR,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    order_count: int = DEFAULT_ORDER_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Generate all three related datasets and return their output paths."""
    if customer_count < 0 or product_count < 0 or order_count < 0:
        raise ValueError("dataset counts must be non-negative")

    customers: list[dict[str, str]] = generate_customers(customer_count, seed)
    products: list[dict[str, str]] = generate_products(product_count, seed)
    orders: list[dict[str, str]] = generate_orders(
        [customer["customer_id"] for customer in customers],
        products,
        order_count,
        seed,
    )
    paths: dict[str, Path] = {
        "customers": output_dir / "customers.csv",
        "products": output_dir / "products.csv",
        "orders": output_dir / "orders.csv",
    }
    write_csv(paths["customers"], customers, CUSTOMER_FIELDS)
    write_csv(paths["products"], products, PRODUCT_FIELDS)
    write_csv(paths["orders"], orders, ORDER_FIELDS)
    LOGGER.info(
        "Generated %d customers, %d products, and %d orders in %s",
        customer_count,
        product_count,
        order_count,
        output_dir,
    )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for dataset generation."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate fake e-commerce CSV datasets."
    )
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMER_COUNT)
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--orders", type=int, default=DEFAULT_ORDER_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the data generator command-line interface."""
    configure_logging()
    args: argparse.Namespace = parse_args(argv)
    generate_dataset(
        output_dir=args.output_dir,
        customer_count=args.customers,
        product_count=args.products,
        order_count=args.orders,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())