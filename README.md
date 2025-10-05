# sales-analysis
Python data pipeline for sales analytics: load CSV → SQLite, clean data, run SQL (SELECT/WHERE/GROUP BY), analyze with Pandas, and visualize insights (top products, revenue, trends). Includes notebook, reproducible scripts, and clear docs. Tech: Python 3.12, sqlite3, Pandas, SQL, Jupyter.

CSV → SQLite Sales Analytics
A practical Python pipeline that loads raw sales CSVs, cleans data with Pandas, persists to SQLite, and runs SQL (SELECT, WHERE, GROUP BY) to surface insights like top‑selling products, monthly revenue, and customer segments, all inside a reproducible Jupyter notebook with charts.

Why this repo
Portfolio‑ready ETL and analytics flow that is small, fast, and easy to reuse across datasets and environments.

Demonstrates end‑to‑end thinking: ingestion, cleaning, storage, querying, analysis, and visualization with clear artifacts.

Features
CSV ingestion → Pandas cleaning → SQLite storage → SQL queries → Pandas analysis → charts in one notebook.

Reusable SQL patterns: SELECT, WHERE, GROUP BY, aggregates, and ready scaffolding for windows.

Minimal dependencies and a deterministic setup for quick cloning and evaluation.

Tech stack
Python 3.x, Pandas, sqlite3, Jupyter Notebook.

GitHub‑Flavored Markdown for docs and clean formatting.

Repository structure
data/ sales.csv (sample or instructions to supply dataset).

src/ load_csv_to_sqlite.py (ingestion + cleaning script).

notebook/ sales_sqlite_analysis.ipynb (queries, EDA, charts).

Quickstart
Create a virtual environment, install dependencies, run the loader, then open the notebook for queries and charts.

Swap in real CSVs by matching column names in the config block or mapping step.

Example queries
Top‑selling products by revenue and quantity using GROUP BY and ORDER BY.

Monthly revenue trends with rolling means for seasonality checks.

Customer segments by order frequency and average order value.

Usage notes
Keep table and column names consistent between Pandas and SQLite to avoid quoting headaches.

For larger CSVs, batch inserts or to_sql chunks can speed up ingest.

SEO‑friendly tips
Use an exact‑match repo name and concise “About” description with primary keywords near the start.

Add Topics: python, pandas, sqlite, sql, etl, data‑engineering, data‑analytics, jupyter‑notebook, csv, data‑cleaning.

Contributing
Open issues for enhancements or bugs, and follow a simple fork‑branch‑PR workflow with clear descriptions.

License
Choose a permissive license (MIT/Apache‑2.0) to maximize reuse and learning.
