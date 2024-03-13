import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data = pd.read_csv('../credit-score/test.csv')

# examine structure of data
print(f'Structure:\n{data.shape}\n')
data.head()

# examine data types
print(f'Data Types\n{data.dtypes}\n')

# check for missing values
print(f'Missing Values\n{data.isnull().sum()}\n')

# filter data
data_filtered = data[data['Payment_of_Min_Amount'] != 'NM']

# Select features relevant to the prediction task
X = data_filtered[['Age', 'Annual_Income',
                   'Num_of_Loan', 'Credit_History_Age']]

# encode the binary target variable 'Payment_of_Min_Amount'
y = data_filtered['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0})

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# instantiate the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)

# evaluate the model
print(f'Accuracy: {accuracy_score(y_test, predictions)}\n')
print(classification_report(y_test, predictions))

# plot the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
