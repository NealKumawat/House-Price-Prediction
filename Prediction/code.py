import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Define the relative path to the CSV file
csv_file = 'Prediction/Data/Bengaluru_House_Data.csv'

# Load the CSV file
df = pd.read_csv(csv_file)


#To drop unwanted columns
df = df.drop(['Availability', 'Society', 'Bath'], axis=1)

# Deletion of duplicate entries
df = df.drop_duplicates()

# ------------------------------------------------------------------------
# To just remain with the numbers
# df['Size'] = df['Size'].str.replace(['^0-9'], '')
# df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
df['Size'] = df['Size'].str.extract(r'(\d+)')

df['Size'] = df['Size'].fillna(0)  # or dropna() if you want to remove them

df['Size'] = df['Size'].astype(int)


df['Total_Sqft'] = pd.to_numeric(df['Total_Sqft'], errors='coerce')

#To fill null values with the meadian of the filled values
df['Total_Sqft'] = df['Total_Sqft'].fillna(df['Total_Sqft'].median())
print((df['Total_Sqft']).dtype)
print(df.shape)

df = df.dropna(subset=['Balcony'])
print(df.shape)

# print(df)
df = pd.get_dummies(df, columns=['Area_Type'], drop_first=True)
print(df)


train = df.drop(['Location', 'Price'], axis = 1)
test = df['Price']

X_train, X_test, Y_train, Y_test = train_test_split(train, test, test_size = 0.3, random_state = 2)
regr = LinearRegression()
regr.fit(X_train, Y_train)

pred = regr.predict(X_test)

print(regr.score(X_test, Y_test))