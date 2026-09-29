# Exploratory Data Analysis (EDA)

## 📌 Project Title

**End-to-End Exploratory Data Analysis Report in Python**

---

## 🎯 Project Objective

The primary objective of this project is to perform a complete Exploratory Data Analysis (EDA) on the Mall Customers dataset. The project demonstrates how raw customer data can be inspected, summarized, visualized, and interpreted to identify meaningful patterns, relationships, distributions, and potential outliers.

---

## ⚠️ Problem Statement

Before building machine-learning models or making data-driven decisions, it is important to understand the structure and quality of a dataset. Customer datasets may contain different data types, unusual values, outliers, and relationships between variables that are not immediately visible.

This project addresses these needs through a structured EDA workflow using statistical analysis and visualization techniques.

---

## 📊 Dataset Description

The project uses the **Mall Customers dataset**, containing **200 customer records**.

### Features

- `CustomerID` — unique customer identifier
- `Gender` — customer gender
- `Age` — customer age
- `Annual Income (k$)` — annual income
- `Spending Score (1-100)` — spending score assigned to the customer

---

## 🔧 Project Workflow

The EDA pipeline follows these major steps:

1. Load the Mall Customers dataset
2. Inspect the dataset structure
3. Generate statistical summaries
4. Check for missing values and duplicates
5. Analyze numerical distributions
6. Detect potential outliers using box plots
7. Compare average values by gender
8. Analyze feature correlations
9. Explore relationships using scatter plots
10. Analyze customer groups by age
11. Summarize key insights and patterns

---

## 📈 Statistical & Visual Analysis

The project includes:

- Statistical summary
- Missing-value and duplicate checks
- Histograms
- Box plots for outlier detection
- Bar charts
- Correlation heatmap
- Scatter plots
- Age-group analysis
- Key insights and patterns

---

## 📊 Charts Generated

The analysis generates the following visualizations:

1. `01_histograms.png` — Distribution of Age, Income, Spending Score + Gender
2. `02_boxplots.png` — Outlier detection
3. `03_bar_charts.png` — Average values by Gender
4. `04_correlation_heatmap.png` — Correlation between features
5. `05_scatter_plots.png` — Relationships between features
6. `06_age_group_analysis.png` — Analysis by age groups

All charts are saved in the `images/` folder.

---

## 📁 Project Structure

```text
eda_project/
├── data/
│   └── Mall_Customers.csv
├── images/
│   ├── 01_histograms.png
│   ├── 02_boxplots.png
│   ├── 03_bar_charts.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_scatter_plots.png
│   └── 06_age_group_analysis.png
├── eda_analysis.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### Run with Python

Open a terminal in the `eda_project` folder.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python eda_analysis.py
```

The generated charts will be saved in the `images/` folder.

### Run with Jupyter Notebook

Install Jupyter if required:

```bash
pip install notebook
```

Start Jupyter:

```bash
python -m notebook
```

Open the included notebook and select:

**Kernel → Restart Kernel and Run All**

---

## 📌 Expected Outcome

After running the project, you will have statistical summaries, data-quality checks, multiple visualizations, and an interpretation of important patterns within the Mall Customers dataset.

---

## 📝 Conclusion

This project demonstrates the complete EDA process and shows how statistics and visualizations can be used to understand customer data before applying machine-learning or segmentation techniques.
