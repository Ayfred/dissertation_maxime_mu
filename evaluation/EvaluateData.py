import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency

# Load datasets
original_data = pd.read_csv('./datasets/data.csv')
generated_data = pd.read_csv('./llama3-8b/results/synthetic_data_llama_3_8b.csv')

# Descriptive statistics comparison
print("Original Data Description:\n", original_data.describe())
print("Generated Data Description:\n", generated_data.describe())

# Categorical distribution comparison
for column in original_data.select_dtypes(include=['object', 'category']):
    print(f"Value counts for {column} in Original Data:\n", original_data[column].value_counts())
    print(f"Value counts for {column} in Generated Data:\n", generated_data[column].value_counts())

# Correlation matrices comparison
print("Original Data Correlation Matrix:\n", original_data.corr())
print("Generated Data Correlation Matrix:\n", generated_data.corr())

# Visualize distribution of a numerical column (Age)
plt.figure(figsize=(10,5))
sns.histplot(original_data['Age'], color='blue', label='Original Data', kde=True)
sns.histplot(generated_data['Age'], color='red', label='Generated Data', kde=True)
plt.legend()
plt.title('Age Distribution Comparison')
plt.show()

# Kolmogorov-Smirnov test for continuous variables
ks_stat, ks_p_value = ks_2samp(original_data['Age'], generated_data['Age'])
print(f"KS Statistic: {ks_stat}, P-value: {ks_p_value}")

# Chi-Square test for categorical variables
contingency_table = pd.crosstab(original_data['Gender'], generated_data['Gender'])
chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2_stat}, P-value: {chi2_p_value}")

# Further steps can include PCA, machine learning model comparison, and more.
