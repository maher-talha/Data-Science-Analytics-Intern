# Data-Science-Analytics-Intern
This repository contains completed tasks for the Data Science &amp; Analytics Internship at DevelopersHub Corporation. Tasks cover core data science skills and regression analysis using Python (pandas, Matplotlib, seaborn, scikit-learn).  **Completed Tasks:** Task 1 (Iris EDA) | Task 3 (Customer Churn Prediction) | Task 4 (Insurance Claim Prediction)
# Task 1: Exploring and Visualizing the Iris Dataset

## Objective
Understand how to read, summarize, and visualize a dataset using Python.
The Iris dataset contains measurements of 150 flowers from 3 species:
Setosa, Versicolor, and Virginica.

## Dataset
- **Name:** Iris Dataset
- **Source:** Scikit-learn built-in dataset
- **Shape:** 150 rows × 5 columns
- **Features:** sepal_length, sepal_width, petal_length, petal_width, species

## Approach
1. Loaded the dataset using `sklearn.datasets` and converted to a `pandas` DataFrame
2. Inspected structure using `.shape`, `.columns`, `.head()`, and `.describe()`
3. Checked for missing values (none found)
4. Created the following visualizations using `matplotlib` and `seaborn`:
   - Scatter plot: Sepal Length vs Sepal Width (by species)
   - Scatter plot: Petal Length vs Petal Width (by species)
   - Histogram: Distribution of all 4 features
   - Box plot: Feature spread and outliers by species
   - Correlation Heatmap: Relationship between features
   - Scatter plot: Sepal Length vs Petal Length

## Results & Insights
- **No missing values** found in the dataset — clean data ready for analysis
- **Petal length and petal width** are highly correlated (r = 0.96)
- **Setosa** is clearly separable from the other two species based on petal size
- **Versicolor and Virginica** overlap slightly in sepal dimensions
- Box plots revealed that **Virginica has the largest petals** on average
- Histograms show that petal features follow a **bimodal distribution**,
  separating Setosa from the rest

## Libraries Used
- `pandas` — data loading and inspection
- `matplotlib` — plotting
- `seaborn` — advanced visualizations
- `scikit-learn` — dataset source
