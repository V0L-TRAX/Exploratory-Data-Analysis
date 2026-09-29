"""
Task 02 - Exploratory Data Analysis (EDA)
=========================================
Complete EDA report for Mall Customers dataset.
Includes statistics, visualizations, and insights.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (Windows-safe)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# --------------------------------------------------
# Setup
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('ggplot')

sns.set_palette("husl")

print("=" * 65)
print("TASK 02 - EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 65)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
print("\n[1] LOADING DATASET")
print("-" * 40)

df = pd.read_csv(DATA_DIR / "Mall_Customers.csv")

print(f"Dataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn names  : {list(df.columns)}")
print("\nFirst 10 rows:")
print(df.head(10))

# --------------------------------------------------
# 2. Basic Information
# --------------------------------------------------
print("\n[2] DATASET INFORMATION")
print("-" * 40)
print(df.info())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing values: {missing.sum()}")

print("\nDuplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# 3. Statistical Summary
# --------------------------------------------------
print("\n[3] STATISTICAL SUMMARY")
print("-" * 40)
print("\nNumerical columns:")
print(df.describe().round(2))

print("\nCategorical columns:")
print(df.describe(include='object'))

print("\nGender value counts:")
print(df['Gender'].value_counts())
print("\nGender percentage:")
print((df['Gender'].value_counts(normalize=True) * 100).round(1))

# --------------------------------------------------
# 4. Univariate Analysis - Histograms
# --------------------------------------------------
print("\n[4] CREATING HISTOGRAMS...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')

axes[0, 0].hist(df['Age'], bins=15, color='#3498db', edgecolor='white', alpha=0.8)
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f"Mean: {df['Age'].mean():.1f}")
axes[0, 0].legend()

axes[0, 1].hist(df['Annual Income (k$)'], bins=15, color='#2ecc71', edgecolor='white', alpha=0.8)
axes[0, 1].set_title('Annual Income Distribution')
axes[0, 1].set_xlabel('Annual Income (k$)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df['Annual Income (k$)'].mean(), color='red', linestyle='--',
                   label=f"Mean: {df['Annual Income (k$)'].mean():.1f}")
axes[0, 1].legend()

axes[1, 0].hist(df['Spending Score (1-100)'], bins=15, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1, 0].set_title('Spending Score Distribution')
axes[1, 0].set_xlabel('Spending Score (1-100)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df['Spending Score (1-100)'].mean(), color='red', linestyle='--',
                   label=f"Mean: {df['Spending Score (1-100)'].mean():.1f}")
axes[1, 0].legend()

# Gender bar chart
gender_counts = df['Gender'].value_counts()
axes[1, 1].bar(gender_counts.index, gender_counts.values, color=['#9b59b6', '#1abc9c'], edgecolor='white')
axes[1, 1].set_title('Gender Distribution')
axes[1, 1].set_xlabel('Gender')
axes[1, 1].set_ylabel('Count')
for i, v in enumerate(gender_counts.values):
    axes[1, 1].text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(str(IMAGES_DIR / "01_histograms.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: images/01_histograms.png")

# --------------------------------------------------
# 5. Box Plots (Outlier Detection)
# --------------------------------------------------
print("\n[5] CREATING BOX PLOTS...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Box Plots - Outlier Detection', fontsize=16, fontweight='bold')

sns.boxplot(y=df['Age'], ax=axes[0], color='#3498db')
axes[0].set_title('Age')

sns.boxplot(y=df['Annual Income (k$)'], ax=axes[1], color='#2ecc71')
axes[1].set_title('Annual Income (k$)')

sns.boxplot(y=df['Spending Score (1-100)'], ax=axes[2], color='#e74c3c')
axes[2].set_title('Spending Score')

plt.tight_layout()
plt.savefig(str(IMAGES_DIR / "02_boxplots.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: images/02_boxplots.png")

# --------------------------------------------------
# 6. Bar Charts - Gender vs Features
# --------------------------------------------------
print("\n[6] CREATING BAR CHARTS...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Average Values by Gender', fontsize=16, fontweight='bold')

gender_age = df.groupby('Gender')['Age'].mean()
axes[0].bar(gender_age.index, gender_age.values, color=['#9b59b6', '#1abc9c'], edgecolor='white')
axes[0].set_title('Average Age by Gender')
axes[0].set_ylabel('Age')
for i, v in enumerate(gender_age.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

gender_income = df.groupby('Gender')['Annual Income (k$)'].mean()
axes[1].bar(gender_income.index, gender_income.values, color=['#9b59b6', '#1abc9c'], edgecolor='white')
axes[1].set_title('Average Income by Gender')
axes[1].set_ylabel('Annual Income (k$)')
for i, v in enumerate(gender_income.values):
    axes[1].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

gender_spend = df.groupby('Gender')['Spending Score (1-100)'].mean()
axes[2].bar(gender_spend.index, gender_spend.values, color=['#9b59b6', '#1abc9c'], edgecolor='white')
axes[2].set_title('Average Spending Score by Gender')
axes[2].set_ylabel('Spending Score')
for i, v in enumerate(gender_spend.values):
    axes[2].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(str(IMAGES_DIR / "03_bar_charts.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: images/03_bar_charts.png")

# --------------------------------------------------
# 7. Correlation Heatmap
# --------------------------------------------------
print("\n[7] CREATING CORRELATION HEATMAP...")

numeric_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
corr = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.3f',
            square=True, linewidths=1, linecolor='white',
            cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(str(IMAGES_DIR / "04_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: images/04_correlation_heatmap.png")

print("\nCorrelation Matrix:")
print(corr.round(3))

# --------------------------------------------------
# 8. Pair Plot / Scatter Plots
# --------------------------------------------------
print("\n[8] CREATING SCATTER PLOTS...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Scatter Plots - Relationships Between Features', fontsize=16, fontweight='bold')

# Age vs Income
for gender, color in zip(['Male', 'Female'], ['#3498db', '#e74c3c']):
    subset = df[df['Gender'] == gender]
    axes[0].scatter(subset['Age'], subset['Annual Income (k$)'],
                    c=color, label=gender, alpha=0.6, s=50, edgecolors='white')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Annual Income (k$)')
axes[0].set_title('Age vs Annual Income')
axes[0].legend()

# Age vs Spending Score
for gender, color in zip(['Male', 'Female'], ['#3498db', '#e74c3c']):
    subset = df[df['Gender'] == gender]
    axes[1].scatter(subset['Age'], subset['Spending Score (1-100)'],
                    c=color, label=gender, alpha=0.6, s=50, edgecolors='white')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Spending Score')
axes[1].set_title('Age vs Spending Score')
axes[1].legend()

# Income vs Spending Score
for gender, color in zip(['Male', 'Female'], ['#3498db', '#e74c3c']):
    subset = df[df['Gender'] == gender]
    axes[2].scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
                    c=color, label=gender, alpha=0.6, s=50, edgecolors='white')
axes[2].set_xlabel('Annual Income (k$)')
axes[2].set_ylabel('Spending Score')
axes[2].set_title('Income vs Spending Score')
axes[2].legend()

plt.tight_layout()
plt.savefig(str(IMAGES_DIR / "05_scatter_plots.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: images/05_scatter_plots.png")

# --------------------------------------------------
# 9. Age Group Analysis
# --------------------------------------------------
print("\n[9] CREATING AGE GROUP ANALYSIS...")

df['Age_Group'] = pd.cut(df['Age'],
                         bins=[0, 25, 35, 45, 55, 100],
                         labels=['18-25', '26-35', '36-45', '46-55', '55+'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Analysis by Age Group', fontsize=16, fontweight='bold')

age_counts = df['Age_Group'].value_counts().sort_index()
axes[0].bar(age_counts.index.astype(str), age_counts.values,
            color=sns.color_palette("husl", 5), edgecolor='white')
axes[0].set_title('Number of Customers by Age Group')
axes[0].set_xlabel('Age Group')
axes[0].set_ylabel('Count')
for i, v in enumerate(age_counts.values):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

age_spend = df.groupby('Age_Group', observed=True)['Spending Score (1-100)'].mean()
axes[1].bar(age_spend.index.astype(str), age_spend.values,
            color=sns.color_palette("husl", 5), edgecolor='white')
axes[1].set_title('Average Spending Score by Age Group')
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Avg Spending Score')
for i, v in enumerate(age_spend.values):
    axes[1].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(str(IMAGES_DIR / "06_age_group_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: images/06_age_group_analysis.png")

# --------------------------------------------------
# 10. Insights Summary
# --------------------------------------------------
print("\n" + "=" * 65)
print("KEY INSIGHTS & PATTERNS")
print("=" * 65)

print("""
1. DATASET OVERVIEW
   • 200 customers with 5 features (CustomerID, Gender, Age, Income, Spending Score)
   • No missing values and no duplicate rows — clean dataset
   • Gender: Female 56% | Male 44%

2. AGE
   • Age ranges from 18 to 70 years
   • Average age ≈ 38.9 years
   • Most customers are between 26–45 years
   • Younger customers (18–35) tend to have higher spending scores

3. ANNUAL INCOME
   • Income ranges from $15k to $137k
   • Average income ≈ $60.6k
   • Distribution is roughly right-skewed (more mid-income customers)
   • Very few customers earn above $100k

4. SPENDING SCORE
   • Score ranges from 1 to 99
   • Average spending score ≈ 50.2 (almost balanced)
   • Distribution is fairly uniform — customers are spread across low to high spenders

5. CORRELATIONS
   • Age and Spending Score have a weak negative correlation
     → Younger people tend to spend more
   • Income and Spending Score have almost no correlation
     → High income does NOT always mean high spending
   • Age and Income have very weak correlation

6. GENDER DIFFERENCES
   • Females slightly outnumber males
   • Average spending score is similar for both genders
   • No strong gender bias in income or spending

7. BUSINESS INSIGHTS
   • Target younger customers (18–35) for high-spending campaigns
   • High-income but low-spending customers are good upselling targets
   • A large group of average-income, average-spending customers exists
     → suitable for standard marketing
   • Dataset is ideal for customer segmentation (clustering)
""")

print("=" * 65)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 65)
print("\nGenerated files in images/ folder:")
print("  01_histograms.png")
print("  02_boxplots.png")
print("  03_bar_charts.png")
print("  04_correlation_heatmap.png")
print("  05_scatter_plots.png")
print("  06_age_group_analysis.png")
print("\nYou can use these charts in your report or presentation.")
