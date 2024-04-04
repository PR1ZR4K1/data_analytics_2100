import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset:
df = pd.read_csv('../customer/Test.csv')


# Task 1: Assess the Size of the Dataset
# ● Identify Number of Rows and Columns:
# ○ Use the shape attribute to find the number of rows (entries) and columns
# (features) in the dataset
rows, columns = df.shape
print(f"The dataset contains {rows} rows and {columns} columns.")

print(df.dtypes)

# check for missing values
missing_values = df.isnull().sum()

# Spits out an object containining the number of missing values for each column
print(f'\nMissing Vals:\n {missing_values}\n')

# Task 1: Assess the Size of the Dataset
# ● Identify Number of Rows and Columns:
# ○ Use the shape attribute to find the number of rows (entries) and columns
# (features) in the dataset


categorical_columns = ['Ever_Married', 'Graduated', 'Profession', 'Var_1']

for column in categorical_columns:

    # Check if the column has missing values
    if df[column].isnull().sum() > 0:

        # Fill missing values with the mode
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)

# # Handling missing values for numerical columns with median

numerical_columns = ['Age', 'Family_Size', 'Work_Experience']

for column in numerical_columns:
    if df[column].isnull().sum() > 0:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

# # check for missing values

missing_values = df.isnull().sum()
print(missing_values)


inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df[['Age', 'Family_Size']])
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['Age', 'Family_Size']])


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Family_Size',
                hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.show()
