#importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np


#function to create dataframes
def create_df(): 
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert date format to datetime
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    return customers_df, products_df, transactions_df

#function to perform exploratory data analysis
def eda(customers_df, products_df, transactions_df):
    
    # Merge datasets
    merged_df = transactions_df.merge(customers_df, on='CustomerID')
    merged_df = merged_df.merge(products_df, on='ProductID')
    merged_df.head(10)

    # Customer eda
    customer_metrics = {
        'total_customers': len(customers_df),
        'customers_by_region': customers_df['Region'].value_counts(),
        'avg_lifetime': (customers_df['SignupDate'].max() - 
                                customers_df['SignupDate'].min()).days
    }
    
    # Product eda
    product_metrics = {
        'total_products': len(products_df),
        'products_by_category': products_df['Category'].value_counts(),
        'avg_price': products_df['Price'].mean(),
        'price_range': (products_df['Price'].min(), products_df['Price'].max())
    }
    
    # Transaction eda
    transaction_metrics = {
        'total_transactions': len(transactions_df),
        'total_revenue': transactions_df['TotalValue'].sum(),
        'avg_value': transactions_df['TotalValue'].mean()
    }
    
    return customer_metrics, product_metrics, transaction_metrics, merged_df

def create_visualizations(merged_df, customers_df, products_df):
    
    # Set style to a built-in style
    plt.style.use('classic')
    
    # 1. Sales by Region
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")  
    ax = sns.barplot(data=merged_df.groupby('Region')['TotalValue'].sum().reset_index(),
                x='Region', y='TotalValue')
    plt.title('Total Sales by Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sales_by_region.png')
    plt.close()
    
    # 2. Product Category Performance
    plt.figure(figsize=(12, 6))
    category_performance = merged_df.groupby('Category').agg({
        'TotalValue': 'sum',
        'TransactionID': 'count'
    }).reset_index()
    
    plt.scatter(category_performance['TransactionID'], 
               category_performance['TotalValue'], 
               s=category_performance['TotalValue']/1000,  # Size scaled down for visibility
               alpha=0.6)
    
    # Add category labels
    for i, category in enumerate(category_performance['Category']):
        plt.annotate(category, 
                    (category_performance['TransactionID'].iloc[i], 
                     category_performance['TotalValue'].iloc[i]))
    
    plt.title('Category Performance: Volume vs Value')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Total Sales Value')
    plt.tight_layout()
    plt.savefig('category_performance.png')
    plt.close()
    
    # 3. Customer Purchase Patterns Over Time
    plt.figure(figsize=(15, 6))
    daily_sales = merged_df.groupby('TransactionDate')['TotalValue'].sum().reset_index()
    plt.plot(daily_sales['TransactionDate'], daily_sales['TotalValue'], 
             linewidth=2, color='blue', alpha=0.7)
    plt.title('Daily Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('sales_trend.png')
    plt.close()

def generate_insights(merged_df, customer_metrics, product_metrics, transaction_metrics):
    
    # 1. Customer Segmentation
    customer_segments = merged_df.groupby('CustomerID').agg({
        'TotalValue': 'sum',
        'TransactionID': 'count'
    }).reset_index()
    
    customer_segments['avg_value'] = (
        customer_segments['TotalValue'] / customer_segments['TransactionID']
    )
    
    # 2. Product Performance
    product_performance = merged_df.groupby('ProductID').agg({
        'TotalValue': 'sum',
        'Quantity': 'sum',
        'TransactionID': 'count'
    }).reset_index()
    
    # 3. Regional Analysis
    regional_analysis = merged_df.groupby('Region').agg({
        'TotalValue': 'sum',
        'CustomerID': 'nunique',
        'TransactionID': 'count'
    }).reset_index()
    
    return {
        'customer_segments': customer_segments,
        'product_performance': product_performance,
        'regional_analysis': regional_analysis
    }

# Main execution
if __name__ == "__main__":
    # Load data
    customers_df, products_df, transactions_df = create_df()
    
    # Perform EDA
    customer_metrics, product_metrics, transaction_metrics, merged_df = eda(
        customers_df, products_df, transactions_df
    )
    
    # Create visualizations
    create_visualizations(merged_df, customers_df, products_df)
    
    # Generate insights
    insights = generate_insights(
        merged_df, customer_metrics, product_metrics, transaction_metrics
    )