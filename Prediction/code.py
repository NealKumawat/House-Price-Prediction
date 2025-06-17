import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


# Define the relative path to the CSV file
csv_file = 'Prediction/Data/Bengaluru_House_Data.csv'

# Load the CSV file
df = pd.read_csv(csv_file)

df = df.drop(['availability', 'society', 'bath'], axis=1)
print(df.shape)

# Deletion of duplicate entries
df = df.drop_duplicates()


# To just remain with the numbers
df['size'] = df['size'].str.replace(['^0-9'], '')
print(df)

