import pandas as pd
import os

# Input and output paths
input_file = "data/supply_chain_data.csv"
output_file = "data/cleaned_supply_chain_data.csv"

print("Loading supply chain dataset...")

# Check whether input file exists
if not os.path.exists(input_file):
    print(f"ERROR: Dataset not found at {input_file}")
    exit()

# Load dataset
df = pd.read_csv(input_file)

print(f"Original dataset shape: {df.shape}")

# Show missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove duplicate rows
duplicates = df.duplicated().sum()
df = df.drop_duplicates()

print(f"\nDuplicates removed: {duplicates}")

# Fill missing numerical values with median
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

# Fill missing categorical values with mode
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

# Create output directory if needed
os.makedirs("data", exist_ok=True)

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print(f"\nCleaned dataset shape: {df.shape}")
print(f"Cleaned dataset saved successfully to: {output_file}")