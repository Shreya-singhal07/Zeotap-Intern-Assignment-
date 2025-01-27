import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

#Calculate DB Index
def davies_bouldin_index(X, labels):
    
    n_clusters = len(np.unique(labels))
    cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
    
    # Calculate cluster dispersions
    dispersions = np.zeros(n_clusters)
    for i in range(n_clusters):
        if sum(labels == i) > 0:  # Check if cluster is not empty
            dispersions[i] = np.mean(cdist(X[labels == i], [cluster_centers[i]]))
    
    # Calculate similarities between clusters
    db_index = 0
    for i in range(n_clusters):
        if sum(labels == i) > 0:  # Check if cluster is not empty
            # Calculate ratios with other clusters
            ratios = np.zeros(n_clusters)
            for j in range(n_clusters):
                if i != j and sum(labels == j) > 0:
                    # Calculate distance between centers
                    center_dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    ratios[j] = (dispersions[i] + dispersions[j]) / center_dist
            # Add maximum ratio for this cluster
            db_index += np.max(ratios[ratios != 0])
    
    return db_index / n_clusters

def create_customer_features(customers_df, transactions_df, products_df):
    
    # Merge transactions with products
    trans_prod = transactions_df.merge(products_df, on='ProductID')
    
    # Calculate customer metrics
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'TotalValue': ['sum', 'mean'],
        'Quantity': ['sum', 'mean']
    }).reset_index()
    
    # Flatten column names
    customer_metrics.columns = ['CustomerID', 'transaction_count', 'total_spend', 
                              'avg_transaction_value', 'total_quantity', 'avg_quantity']
    
    # Calculate RFM metrics
    current_date = transactions_df['TransactionDate'].max()
    
    rfm = transactions_df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (pd.to_datetime(current_date) - pd.to_datetime(x.max())).days,
        'TransactionID': 'count',
        'TotalValue': 'sum'
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
    customer_features = customer_metrics.merge(rfm, on='CustomerID')
    customer_features = customer_features.merge(category_pivot, on='CustomerID')
    
    # Add customer signup age
    signup_age = (pd.to_datetime(current_date) - pd.to_datetime(customers_df['SignupDate'])).dt.days
    customers_df['account_age'] = signup_age
    
    customer_features = customer_features.merge(
        customers_df[['CustomerID', 'Region', 'account_age']], 
        on='CustomerID'
    )
    
    # Convert region to dummy variables
    region_dummies = pd.get_dummies(customer_features['Region'], prefix='region')
    customer_features = pd.concat([customer_features.drop(['Region', 'CustomerID'], axis=1), 
                                 region_dummies], axis=1)
    
    return customer_features

def perform_clustering(features, max_clusters=10):
    """Perform clustering and calculate metrics"""
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Initialize metrics storage
    metrics = {
        'n_clusters': [],
        'db_index': [],
        'silhouette': [],
        'calinski_harabasz': []
    }
    
    # Try different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        
        # Calculate metrics
        metrics['n_clusters'].append(n_clusters)
        metrics['db_index'].append(davies_bouldin_index(scaled_features, labels))
        metrics['silhouette'].append(silhouette_score(scaled_features, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(scaled_features, labels))
    
    return pd.DataFrame(metrics)

#plotting clusters
def visualize_clusters(features, best_n_clusters):
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    
    # Perform clustering with best number of clusters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=labels, cmap='viridis')
    plt.title(f'Customer Segments (n_clusters={best_n_clusters})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig('cluster_visualization.png')
    plt.close()
    
    # Create metrics plot
    metrics_df = perform_clustering(features)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df['n_clusters'], metrics_df['db_index'], marker='o')
    plt.title('Davies-Bouldin Index vs n_clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('DB Index')
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_df['n_clusters'], metrics_df['silhouette'], marker='o')
    plt.title('Silhouette Score vs n_clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig('clustering_metrics.png')
    plt.close()
    
    return labels, metrics_df

# Main function
if __name__ == "__main__":
    # Load data
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Create features
    customer_features = create_customer_features(customers_df, transactions_df, products_df)
    
    # Perform clustering analysis
    metrics_df = perform_clustering(customer_features)
    
    # Find optimal number of clusters (minimum DB Index)
    best_n_clusters = metrics_df.loc[metrics_df['db_index'].idxmin(), 'n_clusters']
    
    # Create visualizations
    labels, metrics = visualize_clusters(customer_features, int(best_n_clusters))
    
    # Print results
    print("\nClustering Results:")
    print(f"Optimal number of clusters: {best_n_clusters}")
    print(f"Best DB Index: {metrics_df['db_index'].min():.3f}")
    print(f"Corresponding Silhouette Score: {metrics_df.loc[metrics_df['db_index'].idxmin(), 'silhouette']:.3f}")