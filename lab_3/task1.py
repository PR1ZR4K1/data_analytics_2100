import numpy as np
import pandas as pd
from warnings import filterwarnings


# Suppress specific FutureWarning from pandas or other libraries
# This is fine because pandas version in requirements.txt still supports
# the methods used in this code
filterwarnings(action='ignore', category=FutureWarning)

cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
        'wheel-base', 'length', 'width', 'height', 'curb-weight',
        'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
        'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
        'city-mpg', 'highway-mpg', 'price'
        ]


# copilot
# data = pd.read_csv('../old-car-data/imports-85.data',
#    names=cols, na_values='?')

data = pd.read_csv('../old-car-data/imports-85.data.txt',
                   names=cols)

print(f'Data Shape:\n{data.shape}')

data = data.replace('?', np.NaN)

print(f'Data Head:\n{data.head()}\n')

# find missing values
print(f'Missing Values:\n{data.isnull().any().any()}\n{data.isnull().sum()}\n')

"""

Next, we want to deal with the missing data.
○ Replace by mean:
■ "normalized-losses": 41 missing data, replace them with mean
■ "stroke": 4 missing data, replace them with mean
■ "bore": 4 missing data, replace them with mean
■ "horsepower": 2 missing data, replace them with mean
■ "peak-rpm": 2 missing data, replace them with mean
○ Replace by frequency:
■ "num-of-doors": 2 missing data, replace them with "four".
■ Reason: 84% sedans is four doors. Since four doors is most frequent, it is
most likely to occur
○ Drop the whole row:
■ "price": 4 missing data, simply delete the whole row
■ Reason: price is what we want to predict. Any data entry without price
data cannot be used for prediction; therefore any row now without price
data is not useful


"""

# Calculate the average of each column.
avg_norm_loss = data['normalized-losses'].astype("float").mean()

print(f'Average Norm Loss:\n{avg_norm_loss}\n')

# Replace "NaN" by mean value in "normalized-losses" column.
data["normalized-losses"].replace(np.NaN, avg_norm_loss, inplace=True)

print(f'Normalized Losses Data:\n{data["normalized-losses"]}\n')

# ● Calculate the mean value for 'bore' column
avg_bore = data["bore"].astype("float").mean()

print(f'Average Bore:\n{avg_bore}\n')


"""

Next, calculate the average bore size of the engines from the provided data, replace any
missing bore sizes with this average (thus handling missing data by imputation), and
then display the updated bore column:

"""

data["bore"].replace(np.NaN, avg_bore, inplace=True)
print(f'Bore Data:\n{data["bore"]}\n')


"""

Next, identify columns with missing data that may need to be addressed through data
cleaning processes such as filling in missing values or removing rows/columns with too
many missing values.

"""
print(f'Data with missing values:\n{data.isnull().sum()}\n')

# Find the average horsepower.

avg_horsepower = data['horsepower'].astype('float').mean(axis=0)

print(f'Average Horsepower:\n{avg_horsepower}\n')

data['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

print(f'Horsepower Data:\n{data["horsepower"]}\n')


# Verify there are no missing values.
missing_horsepower = data['horsepower'].isnull().sum()
print(
    f'Missing values in "horsepower" after replacement: {missing_horsepower}\n')

# Find the most frequent type of doors.

print(f'Most frequent type of doors: {data["num-of-doors"].value_counts()}\n')

# Replace the missing 'num-of-doors' values with the most frequent value.
data['num-of-doors'].replace(np.nan, 'four', inplace=True)

print(f'Num of doors:\n{data["num-of-doors"]}\n')

print(f'Data Head:\n{data.head()}\n')

# Next, let’s address the stroke column.

avg_stroke = data['stroke'].astype('float').mean(axis=0)

print(f'Average Stroke:\n{avg_stroke}\n')

data['stroke'].replace(np.nan, avg_stroke, inplace=True)

# ● Verify no missing values remain in stroke column.
missing_stroke = data['stroke'].isnull().sum()
print("Missing values in 'stroke' after replacement:", missing_stroke)

# Find the average peak rpm.

avg_peak_rpm = data['peak-rpm'].astype('float').mean(axis=0)

print(f'Average Peak RPM:\n{avg_peak_rpm}\n')

# Finally, let’s clean up the missing values from the peak-rpm column.
data['peak-rpm'].replace(np.nan, avg_peak_rpm, inplace=True)

print(f'Peak RPM Data:\n{data["peak-rpm"]}\n')

# ● Verify no missing values remain in peak-rpm column.
missing_peak_rpm = data['peak-rpm'].isnull().sum()
print("Missing values in 'peak-rpm' after replacement:", missing_peak_rpm)
# ● Let’s drop all rows that do not have price data.

before_rows = data.shape[0]
data.dropna(subset=["price"], axis=0, inplace=True)
after_rows = data.shape[0]

print(f"Number of dropped rows {before_rows - after_rows}")

print(f'Price Data:\n{data["price"]}\n')


data.reset_index(drop=True, inplace=True)

print(f'Data Types: \n{data.dtypes}\n')

# Convert data into correct format

data[["bore", "stroke"]] = data[["bore", "stroke"]].astype("float")

data[["normalized-losses"]] = data[["normalized-losses"]].astype("int")

data[["price"]] = data[["price"]].astype("float")

data[["peak-rpm"]] = data[["peak-rpm"]].astype("float")

print(f'Data Head:\n{data.head()}\n')


# Transfomr mpg to L/100km by mathematical operation (235 divided by mpg)

data["highway-mpg"] = 235/data["highway-mpg"]

# Inplace method for pandas is not recommended anymore because upon updates the value of the original data is a copy
# meaning the memory location for the original data is not updated. This is why the inplace method is not recommended
data.rename(columns={"highway-mpg": "highway-L/100km"}, inplace=True)


print(f'Data Head:\n{data.head()}\n')
