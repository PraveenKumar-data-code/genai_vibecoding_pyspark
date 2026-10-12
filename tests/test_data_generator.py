"""Tests for synthetic e-commerce data generation."""

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.data_generator import (
    SyntheticDataGenerator,
    generate_customers,
    generate_dataset,
)


class DataGeneratorTests(unittest.TestCase):
    """Verify generator reproducibility and its CSV outputs."""

    def test_customer_generation_is_reproducible(self) -> None:
        """A fixed seed produces identical fake customer rows."""
        self.assertEqual(generate_customers(5, seed=17), generate_customers(5, seed=17))

    def test_dataset_writes_expected_row_counts(self) -> None:
        """Dataset generation writes all CSVs with headers and requested rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            paths: dict[str, Path] = generate_dataset(
                output_dir=Path(temp_dir),
                customer_count=3,
                product_count=2,
                order_count=7,
                seed=9,
            )

            self.assertEqual(set(paths), {"customers", "products", "orders"})
            for name, expected_count in (
                ("customers", 3),
                ("products", 2),
                ("orders", 7),
            ):
                with paths[name].open("r", newline="", encoding="utf-8") as csv_file:
                    rows: list[dict[str, str]] = list(csv.DictReader(csv_file))
                self.assertEqual(len(rows), expected_count)

    def test_synthetic_generator_returns_expected_frames_and_pareto_share(self) -> None:
        """DataFrames have requested fields and the selected cohort gets 80% of orders."""
        generator: SyntheticDataGenerator = SyntheticDataGenerator(
            num_customers=50,
            num_products=12,
            num_orders=500,
            seed=21,
            reference_date=date(2025, 12, 31),
        )
        customers, products, orders = generator.generate_all()

        self.assertEqual(
            list(customers.columns),
            [
                "customer_id",
                "name",
                "email",
                "age",
                "city",
                "country",
                "registration_date",
            ],
        )
        self.assertEqual(
            list(products.columns),
            ["product_id", "name", "category", "price", "stock", "rating"],
        )
        self.assertEqual(
            list(orders.columns),
            ["order_id", "customer_id", "product_id", "quantity", "order_date"],
        )
        self.assertTrue(customers["age"].between(18, 90).all())
        self.assertTrue(products["price"].between(10, 500).all())
        self.assertTrue(products["rating"].between(1, 5).all())
        self.assertTrue(orders["quantity"].between(1, 10).all())
        high_activity_orders: int = int(
            orders["customer_id"].isin(generator._high_activity_customer_ids).sum()
        )
        self.assertEqual(high_activity_orders, 400)

    def test_zero_sized_generator_keeps_dataframe_schemas(self) -> None:
        """Zero-sized datasets return empty DataFrames with all expected columns."""
        generator: SyntheticDataGenerator = SyntheticDataGenerator(
            num_customers=0,
            num_products=0,
            num_orders=0,
        )
        customers, products, orders = generator.generate_all()

        self.assertEqual(len(customers.columns), 7)
        self.assertEqual(len(products.columns), 6)
        self.assertEqual(len(orders.columns), 5)


if __name__ == "__main__":
    unittest.main()