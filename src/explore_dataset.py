import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load one CSV file from datasets/cic-ids-2018/
dataset_path = Path(__file__).parent.parent / "datasets" / "CSE-CIC-IDS2018" / "02-14-2018.csv"
df = pd.read_csv(dataset_path)

# Strip whitespace from all column names
df.columns = df.columns.str.strip()

# Print the shape of the dataframe
print("Dataset Shape:", df.shape)
print()

# Print all column names
print("Column Names:")
print(df.columns.tolist())
print()

# Replace inf and -inf with NaN, then print total missing value count
df.replace([np.inf, -np.inf], np.nan, inplace=True)
missing_count = df.isna().sum().sum()
print(f"Total Missing Values (including replaced inf/-inf): {missing_count}")
print()

# Print value_counts of the Label column (both count and percentage)
if "Label" in df.columns:
    label_counts = df["Label"].value_counts()
    label_percentages = (df["Label"].value_counts(normalize=True) * 100).round(2)
    
    print("Label Distribution (Count and Percentage):")
    for label in label_counts.index:
        count = label_counts[label]
        percentage = label_percentages[label]
        print(f"  {label}: {count} ({percentage}%)")
    print()

# Print describe() for numeric columns
print("Descriptive Statistics (Numeric Columns):")
print(df.describe())
print()

# Save the label distribution as a bar chart to experiments/label_distribution.png
if "Label" in df.columns:
    label_counts = df["Label"].value_counts()
    
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Label Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / "experiments" / "label_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Label distribution chart saved to: {output_path}")
    plt.close()