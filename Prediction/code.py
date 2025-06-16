import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


# Define the relative path to the CSV file
csv_file = 'data/my_data.csv'

# Load the CSV file
df = pd.read_csv(csv_file)

# Print basic info
print("Column Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nShape of the dataset:")
print(df.shape)
