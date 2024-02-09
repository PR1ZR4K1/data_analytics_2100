import pandas as pd
import matplotlib.pyplot as plot
import seaborn

dataset = pd.read_csv('../online-retail-dataset/Online Retail.csv')

top_countries = dataset.groupby(
    'Country')['InvoiceNo'].nunique().sort_values(ascending=False).head(5)

seaborn.barplot(x=top_countries.index, y=top_countries.values,
                palette='viridis')

plot.title('Top 5 Countries with the Highest Number of Orders')
plot.xlabel('Country')
plot.ylabel('Number of Orders')
plot.xticks(rotation=45)
plot.show()
