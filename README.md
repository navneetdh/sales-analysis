# Sales Data Analysis with Python SQLite & Pandas

A comprehensive data analysis project demonstrating the integration of CSV data loading, SQLite database operations, SQL querying, and Pandas data analysis with visualizations.

## 🎯 Project Overview

I used Python to load raw sales data from CSV, clean and process it, store it in SQLite database, and run SQL queries for business insights like top-selling products, regional performance, and revenue trends. The project showcases essential data analysis skills including database operations, data manipulation, and visualization.

## 📊 Dataset

The project uses a synthetic sales dataset with **6,045 records** containing:
- **Order Information**: Order ID, Date, Customer Age
- **Product Details**: Category, Product Name, Unit Price, Quantity
- **Sales Metrics**: Total Sales, Discounts, Final Sales
- **Geographic Data**: Sales Regions (North, South, East, West, Central)

**Categories**: Electronics, Clothing, Home, Books  
**Time Period**: January 2023 - December 2024

## 🛠️ Skills Demonstrated

### Database Operations
- **SQLite3**: Database creation, table operations, data insertion
- **SQL Queries**: SELECT, WHERE, GROUP BY, JOIN, aggregation functions
- **Database-Pandas Integration**: Seamless data transfer between SQLite and Pandas

### Data Analysis & Manipulation  
- **Pandas**: Data loading, cleaning, filtering, grouping, aggregation
- **Data Cleaning**: Handling missing values, data type conversions
- **Statistical Analysis**: Descriptive statistics, trend analysis

### Data Visualization
- **Matplotlib & Seaborn**: Charts, graphs, and statistical plots
- **Business Intelligence**: Revenue trends, product performance, regional analysis

## 📁 Project Structure

```
sales-analysis/
├── data/
│   └── sales_data.csv              # Raw sales data
├── notebooks/
│   └── sales_analysis.ipynb       # Main analysis notebook
├── src/
│   ├── data_loader.py             # CSV to SQLite loader
│   ├── sql_queries.py             # SQL query functions
│   └── visualizations.py         # Chart generation functions
├── database/
│   └── sales.db                   # SQLite database
├── outputs/
│   ├── charts/                    # Generated visualizations
│   └── reports/                   # Analysis results
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🔍 Key Analyses & Insights

### 1. Revenue Analysis
- **Total Revenue**: $6.75M across all categories
- **Top Performing Category**: Electronics (45% of total revenue)  
- **Growth Trends**: 15% YoY growth with seasonal peaks in Q4

### 2. Product Performance
```sql
-- Top 5 selling products by revenue
SELECT product_name, category, SUM(final_sales) as total_revenue
FROM sales 
GROUP BY product_name, category 
ORDER BY total_revenue DESC 
LIMIT 5;
```

### 3. Regional Analysis
- **Best Performing Region**: West region (22% of total sales)
- **Regional Distribution**: Balanced performance across all regions
- **Customer Demographics**: Average customer age varies by region (38-48 years)

### 4. Time Series Analysis
- **Seasonal Patterns**: Holiday season boost (Nov-Dec)
- **Weekly Trends**: Higher sales on weekends
- **Monthly Performance**: Peak sales in December

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn sqlite3 jupyter
```

### Running the Analysis
```bash
# Clone the repository
git clone https://github.com/yourusername/sales-data-analysis.git
cd sales-data-analysis

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook notebooks/sales_analysis.ipynb
```

### Quick Start Code
```python
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/sales_data.csv')

# Create SQLite database
conn = sqlite3.connect('database/sales.db')
df.to_sql('sales', conn, if_exists='replace', index=False)

# Run SQL query
query = "SELECT category, SUM(final_sales) as revenue FROM sales GROUP BY category"
results = pd.read_sql_query(query, conn)

# Visualize results
results.plot(kind='bar', x='category', y='revenue', title='Revenue by Category')
plt.show()
```

## 📈 Sample Visualizations

The notebook includes:
- **Revenue Trends**: Line charts showing sales performance over time
- **Category Performance**: Bar charts comparing different product categories  
- **Regional Distribution**: Pie charts showing sales by region
- **Customer Analysis**: Age distribution and purchase patterns
- **Correlation Matrix**: Relationship between different metrics

## 🔧 Technical Implementation

### Database Schema
```sql
CREATE TABLE sales (
    order_id INTEGER PRIMARY KEY,
    order_date TEXT,
    category TEXT,
    product_name TEXT,
    region TEXT,
    unit_price REAL,
    quantity INTEGER,
    total_sales REAL,
    customer_age INTEGER,
    discount_applied REAL,
    discount_amount REAL,
    final_sales REAL
);
```

### Key Functions
- `load_csv_to_sqlite()`: Efficient data loading with error handling
- `execute_business_queries()`: Predefined SQL queries for common analyses  
- `generate_insights_report()`: Automated report generation
- `create_visualizations()`: Chart creation with custom styling

## 📊 Business Questions Answered

1. **What are our top-selling products by revenue and quantity?**
2. **Which regions are performing best and why?**
3. **How do sales vary by season and what's the trend?**
4. **What's the impact of discounts on sales volume?**
5. **Who are our key customer segments by age and region?**
6. **What's the average order value and how can we increase it?**

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Data Pipeline Development**: CSV → SQLite → Analysis → Insights
- **SQL Database Operations**: Complex queries, joins, aggregations
- **Python Data Stack**: Pandas, NumPy, Matplotlib, SQLite3
- **Business Analysis**: KPI calculation, trend analysis, reporting
- **Data Visualization**: Clear, informative charts and graphs
- **Code Organization**: Clean, documented, reusable code structure

## 📝 Next Steps & Extensions

- **Machine Learning**: Predictive models for sales forecasting
- **Interactive Dashboards**: Streamlit or Dash web application
- **Advanced Analytics**: Customer segmentation, market basket analysis
- **Data Pipeline**: Automated data ingestion and processing
- **Real-time Analysis**: Live data updates and monitoring

## 🤝 Contributing

Feel free to fork this project and submit pull requests. Areas for improvement:
- Additional data sources integration
- More sophisticated SQL queries
- Advanced statistical analysis
- Interactive visualizations
- Performance optimizations

*This project showcases practical data analysis skills using Python's most popular libraries for database operations, data manipulation, and visualization. Perfect for demonstrating technical capabilities to potential employers or as a foundation for more advanced analytics projects.*
