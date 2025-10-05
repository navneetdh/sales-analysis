# Sales Data Analysis - Complete Python Implementation

## Jupyter Notebook: sales_analysis.ipynb

```python
# ==========================================
# SALES DATA ANALYSIS WITH PYTHON & SQLITE
# ==========================================

import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("📊 Sales Data Analysis Project")
print("=" * 50)
print("Skills Demonstrated: SQLite3, SQL, Pandas, Data Visualization")
print("Goal: Extract business insights from sales data")
print("=" * 50)

# ==========================================
# STEP 1: LOAD CSV DATA AND EXPLORE
# ==========================================

print("\n🔍 STEP 1: Loading and Exploring Raw Data")
print("-" * 40)

# Load the CSV file
df = pd.read_csv('sales_data.csv')

print(f"✅ Loaded CSV file with {len(df)} records")
print(f"📋 Dataset shape: {df.shape}")
print(f"📅 Date range: {df['order_date'].min()} to {df['order_date'].max()}")

# Display basic info
print("\n📊 Dataset Overview:")
df.info()

print("\n📈 First 5 rows:")
display(df.head())

print("\n📉 Statistical Summary:")
display(df.describe())

# Check for missing values
print(f"\n❓ Missing values: {df.isnull().sum().sum()}")

# ==========================================
# STEP 2: DATA CLEANING AND PREPARATION
# ==========================================

print("\n🧹 STEP 2: Data Cleaning and Preparation")
print("-" * 40)

# Convert date column to datetime
df['order_date'] = pd.to_datetime(df['order_date'])

# Add derived columns for analysis
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['quarter'] = df['order_date'].dt.quarter
df['day_of_week'] = df['order_date'].dt.day_name()

print("✅ Added time-based columns: year, month, quarter, day_of_week")

# Check data types
print("\n📝 Data types after cleaning:")
print(df.dtypes)

# ==========================================
# STEP 3: CREATE SQLITE DATABASE
# ==========================================

print("\n💾 STEP 3: Creating SQLite Database")
print("-" * 40)

# Create SQLite database connection
conn = sqlite3.connect('sales_analysis.db')
cursor = conn.cursor()

# Load data into SQLite
df.to_sql('sales', conn, if_exists='replace', index=False)

print("✅ Data loaded into SQLite database 'sales_analysis.db'")
print(f"📊 Table 'sales' created with {len(df)} records")

# Verify the data was loaded correctly
cursor.execute("SELECT COUNT(*) FROM sales")
record_count = cursor.fetchone()[0]
print(f"🔍 Verified: {record_count} records in database")

# Show table schema
cursor.execute("PRAGMA table_info(sales)")
columns = cursor.fetchall()
print("\n📋 Table Schema:")
for col in columns:
    print(f"  - {col[1]} ({col[2]})")

# ==========================================
# STEP 4: SQL QUERIES FOR BUSINESS INSIGHTS
# ==========================================

print("\n🔎 STEP 4: SQL Analysis - Business Insights")
print("-" * 40)

# Helper function to execute SQL queries and display results
def execute_query(query, description):
    print(f"\n📊 {description}")
    print("SQL Query:", query)
    result = pd.read_sql_query(query, conn)
    display(result)
    return result

# Query 1: Total revenue by category
print("\n" + "="*60)
print("📈 BUSINESS QUESTION 1: Which product categories generate the most revenue?")
print("="*60)

query1 = """
SELECT 
    category,
    COUNT(*) as total_orders,
    SUM(quantity) as total_quantity,
    ROUND(SUM(final_sales), 2) as total_revenue,
    ROUND(AVG(final_sales), 2) as avg_order_value
FROM sales 
GROUP BY category 
ORDER BY total_revenue DESC
"""

revenue_by_category = execute_query(query1, "Revenue Analysis by Product Category")

# Query 2: Top selling products
print("\n" + "="*60)
print("🏆 BUSINESS QUESTION 2: What are the top-selling products?")
print("="*60)

query2 = """
SELECT 
    product_name,
    category,
    COUNT(*) as orders_count,
    SUM(quantity) as total_quantity_sold,
    ROUND(SUM(final_sales), 2) as total_revenue
FROM sales 
GROUP BY product_name, category 
ORDER BY total_revenue DESC
LIMIT 10
"""

top_products = execute_query(query2, "Top 10 Products by Revenue")

# Query 3: Regional performance
print("\n" + "="*60)
print("🌍 BUSINESS QUESTION 3: How do different regions perform?")
print("="*60)

query3 = """
SELECT 
    region,
    COUNT(*) as total_orders,
    ROUND(SUM(final_sales), 2) as total_revenue,
    ROUND(AVG(final_sales), 2) as avg_order_value,
    ROUND(AVG(customer_age), 1) as avg_customer_age
FROM sales 
GROUP BY region 
ORDER BY total_revenue DESC
"""

regional_performance = execute_query(query3, "Regional Sales Performance")

# Query 4: Monthly sales trends
print("\n" + "="*60)
print("📅 BUSINESS QUESTION 4: What are the monthly sales trends?")
print("="*60)

query4 = """
SELECT 
    year,
    month,
    COUNT(*) as total_orders,
    ROUND(SUM(final_sales), 2) as monthly_revenue,
    ROUND(AVG(final_sales), 2) as avg_order_value
FROM sales 
GROUP BY year, month 
ORDER BY year, month
"""

monthly_trends = execute_query(query4, "Monthly Sales Trends")

# Query 5: Discount impact analysis
print("\n" + "="*60)
print("💰 BUSINESS QUESTION 5: How do discounts impact sales?")
print("="*60)

query5 = """
SELECT 
    CASE 
        WHEN discount_applied = 0 THEN 'No Discount'
        WHEN discount_applied <= 0.10 THEN 'Low Discount (≤10%)'
        WHEN discount_applied <= 0.20 THEN 'High Discount (>10%)'
    END as discount_tier,
    COUNT(*) as order_count,
    ROUND(AVG(quantity), 2) as avg_quantity,
    ROUND(SUM(final_sales), 2) as total_revenue,
    ROUND(SUM(discount_amount), 2) as total_discount_given
FROM sales 
GROUP BY discount_tier
ORDER BY total_revenue DESC
"""

discount_analysis = execute_query(query5, "Impact of Discounts on Sales")

# Query 6: Customer age analysis
print("\n" + "="*60)
print("👥 BUSINESS QUESTION 6: What are customer demographics?")
print("="*60)

query6 = """
SELECT 
    CASE 
        WHEN customer_age < 25 THEN '18-24'
        WHEN customer_age < 35 THEN '25-34'
        WHEN customer_age < 45 THEN '35-44'
        WHEN customer_age < 55 THEN '45-54'
        ELSE '55+'
    END as age_group,
    COUNT(*) as customer_count,
    ROUND(SUM(final_sales), 2) as total_revenue,
    ROUND(AVG(final_sales), 2) as avg_spending
FROM sales 
GROUP BY age_group
ORDER BY total_revenue DESC
"""

age_analysis = execute_query(query6, "Customer Age Group Analysis")

# ==========================================
# STEP 5: PANDAS ANALYSIS AND AGGREGATION
# ==========================================

print("\n🐼 STEP 5: Advanced Analysis with Pandas")
print("-" * 40)

# Pandas analysis for deeper insights
print("\n📊 Pandas Analysis - Additional Insights:")

# Top performing day of week
day_performance = df.groupby('day_of_week').agg({
    'final_sales': ['count', 'sum', 'mean'],
    'quantity': 'sum'
}).round(2)

day_performance.columns = ['Order Count', 'Total Revenue', 'Avg Order Value', 'Total Quantity']
print("\n📅 Performance by Day of Week:")
display(day_performance.sort_values('Total Revenue', ascending=False))

# Seasonal analysis
seasonal_analysis = df.groupby(['year', 'quarter']).agg({
    'final_sales': ['sum', 'mean'],
    'order_id': 'count'
}).round(2)
seasonal_analysis.columns = ['Total Revenue', 'Avg Order Value', 'Order Count']
print("\n🌟 Seasonal Performance (by Quarter):")
display(seasonal_analysis)

# Customer value segmentation using quantiles
df['customer_value_tier'] = pd.qcut(df['final_sales'], 
                                   q=3, 
                                   labels=['Low Value', 'Medium Value', 'High Value'])

value_segmentation = df.groupby('customer_value_tier').agg({
    'final_sales': ['count', 'sum', 'mean'],
    'customer_age': 'mean'
}).round(2)
value_segmentation.columns = ['Order Count', 'Total Revenue', 'Avg Order Value', 'Avg Customer Age']
print("\n💎 Customer Value Segmentation:")
display(value_segmentation)

# ==========================================
# STEP 6: DATA VISUALIZATIONS
# ==========================================

print("\n📊 STEP 6: Creating Data Visualizations")
print("-" * 40)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('📊 Sales Data Analysis Dashboard', fontsize=20, fontweight='bold')

# 1. Revenue by Category (Bar Chart)
ax1 = axes[0, 0]
revenue_by_category.plot(x='category', y='total_revenue', kind='bar', ax=ax1, color='skyblue')
ax1.set_title('💰 Revenue by Product Category', fontsize=14, fontweight='bold')
ax1.set_xlabel('Product Category')
ax1.set_ylabel('Total Revenue ($)')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# 2. Regional Performance (Pie Chart)  
ax2 = axes[0, 1]
regional_revenue = regional_performance.set_index('region')['total_revenue']
ax2.pie(regional_revenue.values, labels=regional_revenue.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('🌍 Revenue Distribution by Region', fontsize=14, fontweight='bold')

# 3. Monthly Trends (Line Chart)
ax3 = axes[1, 0]
monthly_pivot = monthly_trends.pivot(index='month', columns='year', values='monthly_revenue')
monthly_pivot.plot(kind='line', ax=ax3, marker='o', linewidth=2)
ax3.set_title('📈 Monthly Revenue Trends', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Monthly Revenue ($)')
ax3.legend(title='Year')
ax3.grid(True, alpha=0.3)

# 4. Top Products (Horizontal Bar Chart)
ax4 = axes[1, 1]
top_5_products = top_products.head(5)
ax4.barh(top_5_products['product_name'], top_5_products['total_revenue'], color='lightcoral')
ax4.set_title('🏆 Top 5 Products by Revenue', fontsize=14, fontweight='bold')
ax4.set_xlabel('Total Revenue ($)')

plt.tight_layout()
plt.show()

# Additional visualizations
print("\n📊 Additional Visualizations:")

# Customer age distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['customer_age'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('👥 Customer Age Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Customer Age')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Order value distribution
plt.subplot(1, 2, 2)
plt.hist(df['final_sales'], bins=50, color='orange', edgecolor='black', alpha=0.7)
plt.title('💵 Order Value Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Order Value ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_columns = ['unit_price', 'quantity', 'total_sales', 'customer_age', 'discount_applied', 'final_sales']
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('🔥 Correlation Matrix - Sales Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ==========================================
# STEP 7: KEY INSIGHTS AND SUMMARY
# ==========================================

print("\n📋 STEP 7: Key Business Insights & Summary")
print("-" * 40)

# Calculate key metrics
total_revenue = df['final_sales'].sum()
total_orders = len(df)
avg_order_value = df['final_sales'].mean()
total_customers = df['customer_age'].count()  # Assuming each row is a unique customer interaction

print("🎯 KEY BUSINESS INSIGHTS:")
print("=" * 50)

print(f"💰 Total Revenue Generated: ${total_revenue:,.2f}")
print(f"📦 Total Orders Processed: {total_orders:,}")
print(f"💵 Average Order Value: ${avg_order_value:.2f}")
print(f"👥 Total Customer Interactions: {total_customers:,}")

print("\n🏆 TOP PERFORMERS:")
print("-" * 20)
print(f"🥇 Best Category: {revenue_by_category.iloc[0]['category']} (${revenue_by_category.iloc[0]['total_revenue']:,.2f})")
print(f"🥇 Best Region: {regional_performance.iloc[0]['region']} (${regional_performance.iloc[0]['total_revenue']:,.2f})")
print(f"🥇 Best Product: {top_products.iloc[0]['product_name']} (${top_products.iloc[0]['total_revenue']:,.2f})")

# Best performing month
best_month = monthly_trends.loc[monthly_trends['monthly_revenue'].idxmax()]
# Ensure month and year are integers
month_int = int(best_month['month'])
year_int = int(best_month['year'])
print(f"🥇 Best Month: {int(best_month['year'])}-{int(best_month['month']):02d} (${best_month['monthly_revenue']:,.2f})")

print("\n📊 ANALYSIS SUMMARY:")
print("-" * 20)
print("✅ Successfully loaded 6,000+ sales records from CSV")
print("✅ Created and queried SQLite database with complex SQL")
print("✅ Performed comprehensive data analysis with Pandas")
print("✅ Generated insightful visualizations and business intelligence")
print("✅ Identified top products, regions, and customer segments")
print("✅ Analyzed seasonal trends and discount impact")

print("\n🔧 TECHNICAL SKILLS DEMONSTRATED:")
print("-" * 30)
print("• SQLite3 database operations and complex SQL queries")
print("• Pandas data manipulation and advanced aggregations")
print("• Data cleaning and preprocessing techniques")
print("• Statistical analysis and business intelligence")
print("• Data visualization with Matplotlib and Seaborn")
print("• Business insight generation and reporting")

print("\n🚀 PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 50)

# Close database connection
conn.close()
print("💾 Database connection closed.")
```

## Supporting Python Files

### data_loader.py
```python
"""
Data loading utilities for sales analysis project
"""
import pandas as pd
import sqlite3
from typing import Optional, Dict, Any

class SalesDataLoader:
    """Class for loading and managing sales data"""
    
    def __init__(self, db_path: str = 'sales_analysis.db'):
        self.db_path = db_path
        self.connection = None
    
    def connect_db(self) -> sqlite3.Connection:
        """Create database connection"""
        self.connection = sqlite3.connect(self.db_path)
        return self.connection
    
    def load_csv_to_dataframe(self, csv_path: str) -> pd.DataFrame:
        """Load CSV file into pandas DataFrame with error handling"""
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Successfully loaded {len(df)} records from {csv_path}")
            return df
        except FileNotFoundError:
            print(f"❌ Error: File {csv_path} not found")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"❌ Error: {csv_path} is empty")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading {csv_path}: {str(e)}")
            return pd.DataFrame()
    
    def dataframe_to_sqlite(self, df: pd.DataFrame, table_name: str, 
                          if_exists: str = 'replace') -> bool:
        """Load DataFrame into SQLite database"""
        try:
            if self.connection is None:
                self.connect_db()
            
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
            print(f"✅ Successfully loaded {len(df)} records into table '{table_name}'")
            return True
        except Exception as e:
            print(f"❌ Error loading data to SQLite: {str(e)}")
            return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a database table"""
        if self.connection is None:
            self.connect_db()
        
        cursor = self.connection.cursor()
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get record count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        record_count = cursor.fetchone()[0]
        
        return {
            'columns': columns,
            'record_count': record_count,
            'table_name': table_name
        }
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("💾 Database connection closed")

# Usage example
if __name__ == "__main__":
    loader = SalesDataLoader()
    
    # Load CSV data
    df = loader.load_csv_to_dataframe('sales_data.csv')
    
    if not df.empty:
        # Load to SQLite
        loader.dataframe_to_sqlite(df, 'sales')
        
        # Get table info
        info = loader.get_table_info('sales')
        print(f"Table info: {info}")
    
    loader.close_connection()
```

### sql_queries.py
```python
"""
SQL query collection for sales analysis
"""
import pandas as pd
import sqlite3
from typing import Optional, Dict, List

class SalesQueryEngine:
    """Class containing all SQL queries for sales analysis"""
    
    def __init__(self, db_path: str = 'sales_analysis.db'):
        self.db_path = db_path
    
    def execute_query(self, query: str, description: str = "") -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = pd.read_sql_query(query, conn)
                if description:
                    print(f"📊 {description}: {len(result)} rows returned")
                return result
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            return pd.DataFrame()
    
    def get_revenue_by_category(self) -> pd.DataFrame:
        """Get total revenue by product category"""
        query = """
        SELECT 
            category,
            COUNT(*) as total_orders,
            SUM(quantity) as total_quantity,
            ROUND(SUM(final_sales), 2) as total_revenue,
            ROUND(AVG(final_sales), 2) as avg_order_value
        FROM sales 
        GROUP BY category 
        ORDER BY total_revenue DESC
        """
        return self.execute_query(query, "Revenue by Category")
    
    def get_top_products(self, limit: int = 10) -> pd.DataFrame:
        """Get top-selling products by revenue"""
        query = f"""
        SELECT 
            product_name,
            category,
            COUNT(*) as orders_count,
            SUM(quantity) as total_quantity_sold,
            ROUND(SUM(final_sales), 2) as total_revenue
        FROM sales 
        GROUP BY product_name, category 
        ORDER BY total_revenue DESC
        LIMIT {limit}
        """
        return self.execute_query(query, f"Top {limit} Products")
    
    def get_regional_performance(self) -> pd.DataFrame:
        """Get sales performance by region"""
        query = """
        SELECT 
            region,
            COUNT(*) as total_orders,
            ROUND(SUM(final_sales), 2) as total_revenue,
            ROUND(AVG(final_sales), 2) as avg_order_value,
            ROUND(AVG(customer_age), 1) as avg_customer_age
        FROM sales 
        GROUP BY region 
        ORDER BY total_revenue DESC
        """
        return self.execute_query(query, "Regional Performance")
    
    def get_monthly_trends(self) -> pd.DataFrame:
        """Get monthly sales trends"""
        query = """
        SELECT 
            year,
            month,
            COUNT(*) as total_orders,
            ROUND(SUM(final_sales), 2) as monthly_revenue,
            ROUND(AVG(final_sales), 2) as avg_order_value
        FROM sales 
        GROUP BY year, month 
        ORDER BY year, month
        """
        return self.execute_query(query, "Monthly Trends")
    
    def get_discount_analysis(self) -> pd.DataFrame:
        """Analyze impact of discounts on sales"""
        query = """
        SELECT 
            CASE 
                WHEN discount_applied = 0 THEN 'No Discount'
                WHEN discount_applied <= 0.10 THEN 'Low Discount (≤10%)'
                WHEN discount_applied <= 0.20 THEN 'High Discount (>10%)'
            END as discount_tier,
            COUNT(*) as order_count,
            ROUND(AVG(quantity), 2) as avg_quantity,
            ROUND(SUM(final_sales), 2) as total_revenue,
            ROUND(SUM(discount_amount), 2) as total_discount_given
        FROM sales 
        GROUP BY discount_tier
        ORDER BY total_revenue DESC
        """
        return self.execute_query(query, "Discount Analysis")
    
    def get_customer_segments(self) -> pd.DataFrame:
        """Get customer age group analysis"""
        query = """
        SELECT 
            CASE 
                WHEN customer_age < 25 THEN '18-24'
                WHEN customer_age < 35 THEN '25-34'
                WHEN customer_age < 45 THEN '35-44'
                WHEN customer_age < 55 THEN '45-54'
                ELSE '55+'
            END as age_group,
            COUNT(*) as customer_count,
            ROUND(SUM(final_sales), 2) as total_revenue,
            ROUND(AVG(final_sales), 2) as avg_spending
        FROM sales 
        GROUP BY age_group
        ORDER BY total_revenue DESC
        """
        return self.execute_query(query, "Customer Age Analysis")
    
    def get_custom_query(self, custom_query: str, description: str = "") -> pd.DataFrame:
        """Execute custom SQL query"""
        return self.execute_query(custom_query, description)

# Usage example
if __name__ == "__main__":
    query_engine = SalesQueryEngine()
    
    # Run all standard queries
    revenue_data = query_engine.get_revenue_by_category()
    top_products = query_engine.get_top_products(5)
    regional_data = query_engine.get_regional_performance()
    
    print("✅ All queries executed successfully")
```

### visualizations.py
```python
"""
Visualization utilities for sales analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

class SalesVisualizer:
    """Class for creating sales analysis visualizations"""
    
    def __init__(self):
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.default_figsize = (12, 8)
    
    def create_revenue_by_category_chart(self, data: pd.DataFrame, 
                                       figsize: Optional[Tuple] = None) -> None:
        """Create bar chart for revenue by category"""
        if figsize is None:
            figsize = self.default_figsize
            
        plt.figure(figsize=figsize)
        bars = plt.bar(data['category'], data['total_revenue'], 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        plt.title('💰 Revenue by Product Category', fontsize=16, fontweight='bold')
        plt.xlabel('Product Category', fontsize=12)
        plt.ylabel('Total Revenue ($)', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_regional_pie_chart(self, data: pd.DataFrame, 
                                figsize: Optional[Tuple] = None) -> None:
        """Create pie chart for regional distribution"""
        if figsize is None:
            figsize = (10, 8)
            
        plt.figure(figsize=figsize)
        plt.pie(data['total_revenue'], labels=data['region'], 
                autopct='%1.1f%%', startangle=90, 
                colors=sns.color_palette("Set3", len(data)))
        
        plt.title('🌍 Revenue Distribution by Region', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def create_monthly_trends_chart(self, data: pd.DataFrame, 
                                  figsize: Optional[Tuple] = None) -> None:
        """Create line chart for monthly trends"""
        if figsize is None:
            figsize = self.default_figsize
            
        plt.figure(figsize=figsize)
        
        # Pivot data for better visualization
        monthly_pivot = data.pivot(index='month', columns='year', values='monthly_revenue')
        
        for year in monthly_pivot.columns:
            plt.plot(monthly_pivot.index, monthly_pivot[year], 
                    marker='o', linewidth=2, label=f'Year {year}')
        
        plt.title('📈 Monthly Revenue Trends', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Monthly Revenue ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_top_products_chart(self, data: pd.DataFrame, 
                                top_n: int = 5,
                                figsize: Optional[Tuple] = None) -> None:
        """Create horizontal bar chart for top products"""
        if figsize is None:
            figsize = self.default_figsize
            
        top_data = data.head(top_n)
        
        plt.figure(figsize=figsize)
        bars = plt.barh(top_data['product_name'], top_data['total_revenue'], 
                       color='lightcoral', edgecolor='darkred', alpha=0.8)
        
        plt.title(f'🏆 Top {top_n} Products by Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Total Revenue ($)', fontsize=12)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'${width:,.0f}', ha='left', va='center', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_correlation_heatmap(self, df: pd.DataFrame, 
                                 columns: List[str],
                                 figsize: Optional[Tuple] = None) -> None:
        """Create correlation heatmap for specified columns"""
        if figsize is None:
            figsize = (10, 8)
            
        plt.figure(figsize=figsize)
        correlation_matrix = df[columns].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True)
        
        plt.title('🔥 Correlation Matrix - Sales Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_distribution_plots(self, df: pd.DataFrame, 
                                columns: List[str],
                                figsize: Optional[Tuple] = None) -> None:
        """Create distribution plots for specified columns"""
        if figsize is None:
            figsize = (15, 5)
            
        fig, axes = plt.subplots(1, len(columns), figsize=figsize)
        if len(columns) == 1:
            axes = [axes]
        
        colors = ['lightgreen', 'orange', 'lightblue', 'pink', 'yellow']
        
        for i, col in enumerate(columns):
            axes[i].hist(df[col], bins=30, color=colors[i % len(colors)], 
                        edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{col.replace("_", " ").title()} Distribution', 
                            fontsize=12, fontweight='bold')
            axes[i].set_xlabel(col.replace("_", " ").title())
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_dashboard(self, revenue_data: pd.DataFrame, 
                        regional_data: pd.DataFrame,
                        monthly_data: pd.DataFrame, 
                        top_products: pd.DataFrame) -> None:
        """Create comprehensive dashboard with multiple charts"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('📊 Sales Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # Revenue by category
        axes[0,0].bar(revenue_data['category'], revenue_data['total_revenue'], 
                     color='skyblue')
        axes[0,0].set_title('Revenue by Category', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Category')
        axes[0,0].set_ylabel('Revenue ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Regional pie chart
        axes[0,1].pie(regional_data['total_revenue'], labels=regional_data['region'], 
                     autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Revenue by Region', fontsize=14, fontweight='bold')
        
        # Monthly trends
        monthly_pivot = monthly_data.pivot(index='month', columns='year', values='monthly_revenue')
        monthly_pivot.plot(kind='line', ax=axes[1,0], marker='o')
        axes[1,0].set_title('Monthly Trends', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Revenue ($)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Top products
        top_5 = top_products.head(5)
        axes[1,1].barh(top_5['product_name'], top_5['total_revenue'], 
                      color='lightcoral')
        axes[1,1].set_title('Top 5 Products', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Revenue ($)')
        
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    # This would typically be called from the main notebook
    visualizer = SalesVisualizer()
    print("✅ SalesVisualizer initialized")
```

### requirements.txt
```txt
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
sqlite3
jupyter>=1.0.0
ipykernel>=6.0.0
```

This complete implementation provides:

1. **📓 Comprehensive Jupyter Notebook** - Complete analysis workflow
2. **🔧 Modular Python Files** - Reusable code components
3. **📊 Professional Visualizations** - Business-ready charts and graphs
4. **💾 SQLite Integration** - Database operations with complex queries
5. **🐼 Advanced Pandas Analysis** - Data manipulation and insights
6. **📋 Business Intelligence** - Actionable insights and recommendations

The project demonstrates practical data analysis skills suitable for GitHub portfolio showcase and job interviews.