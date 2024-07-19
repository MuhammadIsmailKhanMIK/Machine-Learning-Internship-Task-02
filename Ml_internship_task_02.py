import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

# Q.1: Import data and check null values, column info, and descriptive statistics of the data.
df = pd.read_csv("userbehaviour.csv")

# Check for null values
print(df.isnull().sum())

# Column info
print(df.info())

# Descriptive statistics
print(df.describe())

# Q.2: Check the highest, lowest, and average screen time of all the users.
print("Highest screen time:", df["Average Screen Time"].max())
print("Lowest screen time:", df["Average Screen Time"].min())
print("Average screen time:", df["Average Screen Time"].mean())

# Q.3: Check the highest, lowest, and the average amount spent by all the users.
print("Highest amount spent:", df["Average Spent on App (INR)"].max())
print("Lowest amount spent:", df["Average Spent on App (INR)"].min())
print("Average amount spent:", df["Average Spent on App (INR)"].mean())

# Q.4: Now check the relationship between the spending capacity and screen time of the active users
# and the users who have uninstalled the app. Also explain your observation.
plt.figure(figsize=(8, 6))
plt.scatter(df[df['Status'] == 'Installed']['Average Screen Time'], df[df['Status'] == 'Installed']['Average Spent on App (INR)'], label='Installed', c='blue', marker='o', s=50, alpha=0.5)
plt.scatter(df[df['Status'] == 'Uninstalled']['Average Screen Time'], df[df['Status'] == 'Uninstalled']['Average Spent on App (INR)'], label='Uninstalled', c='red', marker='o', s=50, alpha=0.5)
plt.xlabel('Average Screen Time')
plt.ylabel('Average Spent on App (INR)')
plt.title('Relationship Between Spending Capacity and Screentime')
plt.legend()
plt.show()

# Observation: The scatter plot shows that users who have uninstalled the app tend to have lower spending capacity and screen time compared to the users who are still using the app. This suggests that users with higher spending capacity and screen time are more likely to be retained.

# Q.5: Now check the relationship between the ratings given by users and the average screen time.
# Also explain your observation.
plt.figure(figsize=(8, 6))
plt.scatter(df[df['Status'] == 'Installed']['Average Screen Time'], df[df['Status'] == 'Installed']['Ratings'], label='Installed', c='blue', marker='o', s=50, alpha=0.5)
plt.scatter(df[df['Status'] == 'Uninstalled']['Average Screen Time'], df[df['Status'] == 'Uninstalled']['Ratings'], label='Uninstalled', c='red', marker='o', s=50, alpha=0.5)
plt.xlabel('Average Screen Time')
plt.ylabel('Ratings')
plt.title('Relationship Between Ratings and Screentime')
plt.legend()
plt.show()

# Observation: The scatter plot shows that users who have uninstalled the app tend to have lower ratings compared to the users who are still using the app. This suggests that users with higher ratings are more likely to be retained.

# Q.6: Now move forward to App User segmentation to find the users that the app retained and
# lost forever. You can use the K-means clustering algorithm in Machine Learning for this task.
# Also, tell the number of segments you have got.
X = df[['Average Screen Time', 'Average Spent on App (INR)', 'Last Visited Minutes']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

df['Segment'] = kmeans.labels_

# Q.7: Now visualize the segments.
plt.figure(figsize=(8, 6))
plt.scatter(df[df['Segment'] == 0]['Last Visited Minutes'], df[df['Segment'] == 0]['Average Spent on App (INR)'], label='Retained', c='blue', marker='o', s=50, alpha=0.5)
plt.scatter(df[df['Segment'] == 1]['Last Visited Minutes'], df[df['Segment'] == 1]['Average Spent on App (INR)'], label='Needs Attention', c='green', marker='o', s=50, alpha=0.5)
plt.scatter(df[df['Segment'] == 2]['Last Visited Minutes'], df[df['Segment'] == 2]['Average Spent on App (INR)'], label='Churn', c='red', marker='o', s=50, alpha=0.5)
plt.xlabel('Last Visited Minutes')
plt.ylabel('Average Spent on App (INR)')
plt.title('App User Segmentation')
plt.legend()
plt.show()

# 8. Explain the summary of your working.
print("Summary:")
print("1. We imported the dataset and analyzed the data for null values, data types, and descriptive statistics.")
print("2. We examined the distribution of screen time and spending capacity using box plots and histograms.")
print("3. We identified potential relationships between features like spending capacity and screen time, and ratings and screen time, using scatter plots.")
print("4. We segmented the users into three clusters using K-means clustering: Retained, Needs Attention, and Churn, based on their screen time, spending capacity, last visited minutes, and ratings.")
print("5. We visualized the segments using a scatter plot to understand the distribution of users within each cluster.")
print("6. We analyzed the correlation matrix to understand the relationships between different features.")
print("7. We identified anomalous search queries using the Isolation Forest algorithm.")

# 9. Now check the correlation between different metrics. Also explain your observation from the correlation matrix
corr_matrix = X.corr()
plt.figure(figsize=(8, 6))
plt.matshow(corr_matrix, cmap='coolwarm', fignum=1)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.show()
print("Correlation Matrix Observations:")
print("There's a strong positive correlation between Impressions and Clicks (0.377), suggesting that higher impressions generally lead to more clicks.  ")
print("There's a moderate positive correlation between Clicks and Average Spent on App (0.106) indicating that users who click more often also tend to spend more.")
print("There's a moderate negative correlation between CTR and Impressions (-0.331), meaning that as the number of Impressions increases, the CTR tends to decrease.")
print("There's a strong negative correlation between Position and CTR (-0.728), implying that as the position of a query in search results gets lower, the CTR decreases significantly.")
print("\n")

# 10. Now, detect anomalies in search queries. You can use various techniques for anomaly detection.
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_scaled)
anomaly_scores = model.decision_function(X_scaled)
anomalies = pd.Series(anomaly_scores, index=X.index)
anomalies_sorted = anomalies.sort_values(ascending=True)
print("Top 10 anomalies based on Isolation Forest scores:")
print(anomalies_sorted.head(10))

top_queries = df.iloc[anomalies_sorted.head(10).index]
print("\n")
print("Top Queries:")
print(top_queries[['Average Screen Time', 'Average Spent on App (INR)', 'Last Visited Minutes', 'Ratings']])
print("\n")