# E-commerce Data Pipeline

Generate reproducible fake customer, product, and order data, then use PySpark to produce business insight tables.

## Requirements

- Python 3.10 or later
- Java 17 or later for PySpark

Install Python dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the pipeline

Generate 1,000 customers, 100 products, and 5,000 orders:

```powershell
python -m src.data_generator
```

Choose dataset sizes or a different output directory:

```powershell
python -m src.data_generator --customers 250 --products 30 --orders 1200 --seed 7
python -m src.data_generator --output-dir data/raw
```

Analyze the generated CSV files with local Spark:

```powershell
python -m src.spark_analytics
```

The reusable `SalesAnalytics` class also loads Parquet data and provides
top-customer revenue, category sales, and monthly revenue-growth DataFrames:

```python
from src.spark_analytics import SalesAnalytics

analytics = SalesAnalytics()
spark = analytics.create_spark_session()
try:
    orders = analytics.load_parquet("data/raw/orders.parquet")
    products = analytics.load_parquet("data/raw/products.parquet")
    top_customers = analytics.top_customers_by_revenue(orders, products)
    category_sales = analytics.sales_by_category(orders, products)
    monthly_trends = analytics.monthly_trends(orders, products)
finally:
    spark.stop()
```

Generate the configured customer, product, and order DataFrames as Parquet files:

```powershell
python main.py
```

This writes `customers.parquet`, `products.parquet`, and `orders.parquet` under
`data/raw/` and reports the generation time and each file's size. Parquet output
uses the `pyarrow` dependency.

Run all Parquet-based analytics and display the results with operation timings:

```powershell
python run_analytics.py
```

Spark writes `monthly_sales`, `top_products`, and `customer_spend` as CSV output directories under `data/processed/`. Revenue metrics include shipped and completed orders, excluding pending and cancelled orders. The output directories contain Spark part files and can be read as CSV directories.

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

## DataFrame generator

`SyntheticDataGenerator` returns pandas DataFrames and defaults to 100,000 customers, 10,000 products, and 1,000,000 orders. Its Pareto-weighted customer cohorts allocate 80% of orders to the top 20% of customers.

```python
from src.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator()
customers, products, orders = generator.generate_all()
```

The command-line generator above retains the smaller CSV defaults used by the Spark analytics stage.

## Project structure

```text
.
|-- data/
|   |-- processed/       # Spark insight tables
|   `-- raw/             # Generated CSV and Parquet datasets
|-- notebooks/           # Jupyter notebook workspace
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_generator.py
|   `-- spark_analytics.py
|-- tests/
|   `-- test_data_generator.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Data outputs

`python -m src.data_generator` writes `customers.csv`, `products.csv`, and
`orders.csv` under `data/raw/` by default. `python main.py` writes the
DataFrame-based `customers.parquet`, `products.parquet`, and `orders.parquet`
there. The complete dataset is reproducible for a fixed seed, and generated
personal details are fictional. Spark analysis produces:

- `monthly_sales`: order count and revenue grouped by month
- `top_products`: units sold and revenue grouped by product
- `customer_spend`: order count and spend grouped by customer

Set `LOG_LEVEL` (for example, `LOG_LEVEL=DEBUG`) to adjust application logging.