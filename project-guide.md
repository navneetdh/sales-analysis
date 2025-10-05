# Sales Data Analysis Project

## Project Structure
```
sales-data-analysis/
├── data/
│   └── sales_data.csv              # Generated sales dataset (6,045 records)
├── notebooks/
│   └── sales_analysis.ipynb       # Main analysis notebook
├── src/
│   ├── data_loader.py             # CSV to SQLite utilities
│   ├── sql_queries.py             # SQL query functions
│   └── visualizations.py         # Chart generation
├── outputs/
│   ├── sales_analysis_demo.png    # Sample visualizations
│   └── sales_demo.db              # SQLite database
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

## Quick Start Guide

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv sales_env
source sales_env/bin/activate  # On Windows: sales_env\Scripts\activate

# Install dependencies
pip install pandas sqlite3 matplotlib seaborn jupyter numpy
```

### 2. Run the Analysis
```bash
# Start Jupyter notebook
jupyter notebook notebooks/sales_analysis.ipynb

# Or run the demo script
python src/demo.py
```

### 3. Key Files to Review
- `notebooks/sales_analysis.ipynb` - Complete analysis with SQL queries and visualizations
- `data/sales_data.csv` - Sample dataset with 6,045 sales records
- `outputs/sales_analysis_demo.png` - Generated charts showing key insights

## Skills Demonstrated

### Technical Skills
✅ **SQLite3 Database Operations**
- Database creation and table management
- Complex SQL queries (SELECT, WHERE, GROUP BY, JOIN)
- Data insertion and retrieval

✅ **SQL Query Expertise**
- Aggregation functions (SUM, COUNT, AVG)
- Conditional logic (CASE statements)  
- Sorting and filtering
- Subqueries and data grouping

✅ **Pandas Data Analysis**
- CSV loading and data cleaning
- DataFrame manipulation and filtering
- Grouping and aggregation operations
- Time series analysis
- Statistical calculations

✅ **Data Visualization**
- Matplotlib charts and graphs
- Seaborn statistical plots
- Business dashboard creation
- Trend analysis visualization

### Business Analysis Skills
✅ **Revenue Analysis** - Identifying top-performing categories and products
✅ **Customer Segmentation** - Age-based analysis and regional performance
✅ **Trend Analysis** - Monthly and seasonal sales patterns
✅ **Performance Metrics** - KPIs, averages, and growth calculations

## Sample Results

### Key Business Insights
- **Total Revenue Generated**: $6,745,928.75
- **Average Order Value**: $116.18  
- **Top Performing Category**: Electronics (45% of revenue)
- **Best Region**: West region (22% of sales)
- **Peak Sales Period**: December 2024

### SQL Query Examples
```sql
-- Top selling products by revenue
SELECT product_name, category, 
       SUM(final_sales) as total_revenue
FROM sales 
GROUP BY product_name, category 
ORDER BY total_revenue DESC 
LIMIT 5;

-- Monthly sales trends
SELECT year, month, 
       COUNT(*) as orders,
       SUM(final_sales) as revenue
FROM sales 
GROUP BY year, month 
ORDER BY year, month;

-- Customer age analysis by region
SELECT region, 
       COUNT(*) as customers,
       AVG(customer_age) as avg_age,
       SUM(final_sales) as revenue
FROM sales 
GROUP BY region;
```

### Pandas Analysis Examples
```python
# Revenue by category using Pandas
category_revenue = df.groupby('category')['final_sales'].agg(['sum', 'count', 'mean'])

# Time series analysis
df['order_date'] = pd.to_datetime(df['order_date'])
monthly_trends = df.groupby(df['order_date'].dt.to_period('M'))['final_sales'].sum()

# Customer segmentation
age_groups = pd.cut(df['customer_age'], bins=[0, 25, 35, 45, 55, 100], 
                   labels=['18-25', '26-35', '36-45', '46-55', '55+'])
df['age_group'] = age_groups
segment_analysis = df.groupby(['age_group', 'category'])['final_sales'].sum()
```

## Project Highlights

### Data Pipeline
1. **CSV Data Loading** → Load raw sales data with error handling
2. **Data Cleaning** → Process dates, handle missing values, add derived columns  
3. **SQLite Storage** → Create database schema and load processed data
4. **SQL Analysis** → Execute business queries for insights
5. **Pandas Analysis** → Advanced data manipulation and statistical analysis
6. **Visualization** → Create charts and business dashboards
7. **Reporting** → Generate insights and recommendations

### Technical Architecture
- **Data Layer**: CSV files and SQLite database
- **Processing Layer**: Python scripts with Pandas
- **Analysis Layer**: SQL queries and statistical functions
- **Presentation Layer**: Jupyter notebooks with visualizations
- **Output Layer**: Charts, reports, and business insights

This project demonstrates practical data analysis skills essential for data analyst and business intelligence roles, showcasing proficiency with industry-standard tools and real-world business scenarios.