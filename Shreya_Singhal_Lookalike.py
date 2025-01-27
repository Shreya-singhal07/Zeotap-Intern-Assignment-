import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

#function to load data
def load_and_preprocess_data():

    # Load datasets
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert dates
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    return customers_df, products_df, transactions_df

#adding new features
def create_customer_features(customers_df, transactions_df, products_df):
    
    # Merge transactions with products to get category information
    trans_prod = transactions_df.merge(products_df, on='ProductID')
    
    # Calculate customer transaction features
    customer_features = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',  # Number of transactions
        'TotalValue': ['sum', 'mean'],  # Total spend and average transaction value
        'Quantity': ['sum', 'mean']  # Total quantity and average quantity per transaction
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = ['CustomerID', 'num_transactions', 'total_spend', 
                               'avg_transaction_value', 'total_quantity', 'avg_quantity']
    
    # Calculate recency, frequency, monetary (RFM) scores
    current_date = transactions_df['TransactionDate'].max()
    
    rfm = transactions_df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (current_date - x.max()).days,  # R
        'TransactionID': 'count',  # F
        'TotalValue': 'sum'  # M
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'recency', 'frequency', 'monetary']
    
    # Calculate category preferences
    category_pivot = pd.pivot_table(
        trans_prod,
        values='Quantity',
        index='CustomerID',
        columns='Category',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Merge all features
    customer_features = customer_features.merge(rfm, on='CustomerID')
    customer_features = customer_features.merge(category_pivot, on='CustomerID')
    
    # Add customer signup age
    signup_age = (current_date - customers_df['SignupDate']).dt.days
    customers_df['account_age'] = signup_age
    
    customer_features = customer_features.merge(
        customers_df[['CustomerID', 'Region', 'account_age']], 
        on='CustomerID'
    )
    
    # Convert region to dummy variables
    region_dummies = pd.get_dummies(customer_features['Region'], prefix='region')
    customer_features = pd.concat([customer_features.drop('Region', axis=1), 
                                 region_dummies], axis=1)
    
    return customer_features

#functions for Lookalike Model implementation
def calculate_similarity_scores(customer_features):
 
    # Select features for similarity calculation
    feature_cols = customer_features.columns.difference(['CustomerID'])
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_features[feature_cols])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(scaled_features)
    
    return similarity_matrix, customer_features['CustomerID'].values

def get_top_lookalikes(customer_id, similarity_matrix, customer_ids, n=3):
  
    # Find index of customer
    customer_idx = np.where(customer_ids == customer_id)[0][0]
    
    # Get similarity scores for this customer
    customer_similarities = similarity_matrix[customer_idx]
    
    # Get indices of top N similar customers (excluding self)
    similar_indices = np.argsort(customer_similarities)[::-1][1:n+1]
    
    # Get customer IDs and similarity scores
    lookalikes = [
        (customer_ids[idx], customer_similarities[idx])
        for idx in similar_indices
    ]
    
    return lookalikes

def create_lookalike_mapping(customer_features, output_file='Lookalike.csv'):
    
    # Calculate similarity scores
    similarity_matrix, customer_ids = calculate_similarity_scores(customer_features)
    
    # Generate mappings for first 20 customers
    mappings = []
    for i in range(20):
        customer_id = customer_ids[i]
        lookalikes = get_top_lookalikes(customer_id, similarity_matrix, customer_ids)
        
        # Format lookalike string
        lookalike_str = ', '.join([
            f"{cid}({score:.3f})"
            for cid, score in lookalikes
        ])
        
        mappings.append({
            'CustomerID': customer_id,
            'Lookalikes': lookalike_str
        })
    
    # Create and save mapping DataFrame
    mapping_df = pd.DataFrame(mappings)
    mapping_df.to_csv(output_file, index=False)
    
    return mapping_df

# Main function
if __name__ == "__main__":
    # Load data
    customers_df, products_df, transactions_df = load_and_preprocess_data()
    
    # Create customer features
    customer_features = create_customer_features(customers_df, transactions_df, products_df)
    
    # Generate lookalike mapping
    lookalike_mapping = create_lookalike_mapping(customer_features)
    
    # Print first few mappings
    print(lookalike_mapping.head())