# Task 02 – Exploratory Data Analysis (EDA)

## Project Overview
This project creates a complete **Exploratory Data Analysis report** for the Mall Customers dataset.

It includes:
- Statistical summary
- Missing value & duplicate check
- Histograms
- Box plots (outlier detection)
- Bar charts
- Correlation heatmap
- Scatter plots
- Age group analysis
- Key insights and patterns

---

## How to Run

1. Open terminal in the `eda_project` folder
2. Install dependencies (only once):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python eda_analysis.py
   ```

All charts will be saved in the `images/` folder.

---

## Project Structure
```
eda_project/
├── data/
│   └── Mall_Customers.csv
├── images/                  ← charts saved here
├── eda_analysis.py          ← main script
├── requirements.txt
└── README.md
```

---

## Charts Generated
1. `01_histograms.png` – Distribution of Age, Income, Spending Score + Gender
2. `02_boxplots.png` – Outlier detection
3. `03_bar_charts.png` – Average values by Gender
4. `04_correlation_heatmap.png` – Correlation between features
5. `05_scatter_plots.png` – Relationships between features
6. `06_age_group_analysis.png` – Analysis by age groups

---

## Dataset
- **Mall Customers** (200 customers)
- Features: CustomerID, Gender, Age, Annual Income, Spending Score
