import pandas as pd

# Load the dataset
netflix = pd.read_csv('../netflix/shows.csv')

netflix.head()

numeric_data = netflix[['Year', 'IMDB_Rating', 'Netflix']]
correlations = numeric_data.corr()

print(correlations)
