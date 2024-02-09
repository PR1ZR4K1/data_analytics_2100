import pandas as pd
import matplotlib.pyplot as plot
import seaborn

dataset = pd.read_csv('../online-retail-dataset/Online Retail.csv')

seaborn.scatterplot(data=dataset, x='Quantity', y='UnitPrice', alpha=0.5)

plot.title('Relationship Between Quantity and UnitPrice')
plot.xlabel('Quantity')
plot.ylabel('UnitPrice')
plot.xscale('log')  # Optional, based on data distribution
plot.yscale('log')  # Optional, based on data distribution
plot.show()
