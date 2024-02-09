import pandas as pd
import matplotlib.pyplot as plot
import seaborn


dataset = pd.read_csv('../online-retail-dataset/Online Retail.csv')

seaborn.histplot(dataset['UnitPrice'], bins=30, kde=True)
plot.title('Distribution of Unit Prices')
plot.xlabel('Unit Price')
plot.ylabel('Frequency')
plot.show()
