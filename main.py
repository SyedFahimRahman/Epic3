import pandas as pd
from scipy import stats

# Provide the full or relative path to your CSV file
file_path = 'C:/Users/23376066/PycharmProjects/Epic3/datasets/raw/terrorist-attacks new.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values:\n", missing_values)

# Impute missing values (replace with mean for numeric columns)
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# One-hot encoding for categorical column 'Entity'
df_encoded = pd.get_dummies(df, columns=['Entity'])

# Identify outliers using z-score for the 'Terrorist attacks' column
z_scores = stats.zscore(df['Terrorist attacks'])

# Define threshold for outlier detection
threshold = 3

# Filter outliers
outliers = (z_scores > threshold) | (z_scores < -threshold)
df_filtered = df[~outliers].copy()

# Drop rows with missing values after outlier removal
df_filtered.dropna(inplace=True)

# Display the filtered DataFrame
print("\nFiltered DataFrame after preprocessing:")
print(df_filtered.head())
