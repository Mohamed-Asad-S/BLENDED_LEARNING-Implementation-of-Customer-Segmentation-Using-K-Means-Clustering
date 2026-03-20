# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data: Import your dataset.

2.Explore Data: Check for initial patterns and distributions.

3.Select Features: Choose the most important variables for grouping.

4.Preprocess Data: Clean and scale the information so it's ready for the algorithm.

5.Find Optimal Clusters: Determine the best number of groups to create (e.g., using the Elbow method).

6.Train Model: Run K-Means to automatically sort the data into those groups.

7.Analyze & Visualize: Graph the final clusters to understand the differences between them.

## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: Mohamed Asad S
RegisterNumber: 212225040238

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent MKL memory leak warning on Windows
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

# Optional: suppress only the specific warning message if you want cleaner output
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

# Step 1: Load the dataset
data = pd.read_csv("CustomerData.csv")

# Step 2: Explore the data
print(data.head())
print(data.columns)
print("="*50)
print("Name: Mohamed Asad S")
print("Reg. No: 212225040238")
print("="*50)
# Step 3: Select relevant features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Step 4: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Elbow method to find optimal clusters
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicit n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Step 6: Train KMeans with chosen clusters
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)  # Explicit n_init
kmeans.fit(X_scaled)

# Step 7: Add cluster labels to data
data['Cluster'] = kmeans.labels_

# Silhouette score
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# Step 8: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot
  (
    data=data,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100,
    alpha=0.7
  )

# Plot cluster centroids (inverse scale)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.75, marker='X', label='Centroids')

plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
*/
```

## Output:
<img width="725" height="264" alt="Screenshot 2026-03-21 004702" src="https://github.com/user-attachments/assets/d6d97f24-6cb2-4eb2-a3a9-25ffbca08dda" />
<img width="882" height="505" alt="Screenshot 2026-03-21 004715" src="https://github.com/user-attachments/assets/a4866301-dc30-468b-ba19-1f748d0a8e0f" />
<img width="990" height="636" alt="Screenshot 2026-03-21 004724" src="https://github.com/user-attachments/assets/acfaacc9-1eab-47f7-91d9-75c288b59383" />


## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
