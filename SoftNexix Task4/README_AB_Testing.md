# E-Commerce A/B Testing Analysis

This project performs a **complete A/B Testing and Hypothesis Testing workflow** using an e-commerce dataset.  
It follows the steps outlined in the _"Data Analysis Using Python – Task 4"_ internship assignment.

---

## Project Overview

The goal of this project is to analyze the performance of a **new webpage design** compared to an **old design** by testing whether the new version improves conversion rates.

Dataset used: `ab_data.csv`

---

## Tasks Performed

### 1️ Data Loading and Exploration

### 2 Data Cleaning

### 3 Hypothesis Formulation

**Null Hypothesis (H₀):** Conversion rate of new page ≤ old page  
**Alternative Hypothesis (H₁):** Conversion rate of new page > old page

### 4 Two-Proportion Z-Test

- Compared conversion rates between `control` and `treatment` groups.
- Calculated **Z-score** and **p-value** to determine statistical significance.

### 5 Confidence Interval Visualization

- Calculated 95% confidence intervals for both groups.
- Visualized conversion rate intervals using Matplotlib.

### 6 Advanced Statistical Tests (Conditional)

- **Chi-Square Test**: Checks if categorical variable (e.g., `device`) affects conversion rate.  
  _(Skipped automatically if the dataset lacks `device` column.)_
- **T-Test**: Checks if continuous metric (e.g., `session_duration`) differs between groups.  
  _(Skipped automatically if the dataset lacks `session_duration` column.)_

### 7 Power Analysis (Sample Size Calculation)

### 8 Business Insights

---

## Results Summary

| Metric          | Control | Treatment |
| --------------- | ------- | --------- |
| Conversion Rate | ~7.5%   | ~9.0%     |
| Z-score         | ~4.7    | p < 0.001 |

**Interpretation:** Since p < 0.05, we reject H₀.  
 The new design performs significantly better than the old one.
