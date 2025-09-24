import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from scipy import stats


customerEcomm = pd.read_csv('olist_customers_dataset.csv')
print(customerEcomm.head().to_string())
print("\n")
print(customerEcomm.info())
print("\n")
print(customerEcomm.shape)


print("Data Shape:", customerEcomm.shape)
print("\nData Types:\n", customerEcomm.dtypes)
print("\nMissing Values:\n", customerEcomm.isnull().sum())
print("\nSummary Statistics of Braxllians Commerece Data:\n", customerEcomm.describe(include='all'))


# 3. Handle Missing Values (Simple Imputation for EDA)
print("Since the Data types of all the Coloumns in the DataFrame are object type, we can fill the missing values with the mode of each column.")


if customerEcomm['customer_zip_code_prefix'].isnull().sum() > 0:
    customerEcomm['customer_zip_code_prefix'].fillna(customerEcomm['customer_zip_code_prefix'].mode()[0], inplace=True)
if customerEcomm['customer_city'].isnull().sum() > 0:
    customerEcomm['customer_city'].fillna(customerEcomm['customer_city'].mode()[0], inplace=True)
if customerEcomm['customer_state'].isnull().sum() > 0:
    customerEcomm['customer_state'].fillna(customerEcomm['customer_state'].mode()[0], inplace=True)

print("\n The Dataframe after handling the missing values:\n", customerEcomm.head().to_string())


# 4. Distribution Analysis
print("\n For the Distribution Analysis I am Using both histogram and boxplot to visualize the distribution of the numerical column 'customer_zip_code_prefix'. ")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(customerEcomm['customer_zip_code_prefix'], bins=30, kde=True, ax=ax[0])
ax[0].set_title('Customer Zip Code Prefix Distribution')
sns.boxplot(x=customerEcomm['customer_zip_code_prefix'], ax=ax[1])
ax[1].set_title('Boxplot: Zip Code Prefix')
plt.tight_layout()
plt.show()


# Categorical Features (Countplots)
print("For the Categorical Features I am using Countplots to visualize the distribution of 'customer_state' and 'customer_city'.")
plt.figure(figsize=(10, 5))
sns.countplot(x='customer_state', data=customerEcomm, order=customerEcomm['customer_state'].value_counts().index)
plt.title('Customers by State')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 5))
citiesvijselc = customerEcomm['customer_city'].value_counts().head(20).index
sns.countplot(y='customer_city', data=customerEcomm[customerEcomm['customer_city'].isin(citiesvijselc)],
              order=citiesvijselc)
plt.title('Top 20 Customer Cities Observed After Careful Analysis is : ')
plt.show()


# 5. Correlation & Relationships
# Correlation Matrix (Heatmap)
print("For Correlation & Relationships I am using a Heatmap to visualize the correlation matrix of numerical features.")
plt.figure(figsize=(10, 6))
sns.heatmap(customerEcomm.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Cross-Tabulation (example: state vs. unique customers)
state_unique = pd.crosstab(customerEcomm['customer_state'], customerEcomm['customer_unique_id'])
print("\nState vs Unique ID Crosstab:\n", (state_unique.iloc[:, :10].head().to_string()))  


# Cross-Tabulation (example: state vs. unique customers)
print("Here i am using Cross-Tabulation to see the relationship between 'customer_state' and 'customer_unique_id'.")
state_unique = customerEcomm.groupby('customer_state')['customer_unique_id'].nunique()
print("\nUnique Customers per State:\n", state_unique.head())

# 6. Outlier Detection
#Boxplot for ZIP Outliers
print("For Outlier Detection I am using Boxplots and Z-Score Analysis to identify outliers in the numerical column 'customer_zip_code_prefix'.")
plt.figure(figsize=(8, 4))
sns.boxplot(x=customerEcomm['customer_zip_code_prefix']).set_title('Zip Code Outliers')
plt.show()

# Z-Score Analysis for Zip Code Prefix (numerical column)
zsrc_cust = np.abs(stats.zscore(customerEcomm['customer_zip_code_prefix']))
outliers = customerEcomm[zsrc_cust > 3]
print(f"Found {len(outliers)} outliers in Zip Code Prefix")

# 7. Advanced Visualizations
# Faceted Analysis – example: by state for zip distribution
print("For Advanced Visualizations I am using FacetGrid Analysis to visualize the distribution of 'customer_zip_code_prefix' across different 'customer_state'.")
gvth = sns.FacetGrid(customerEcomm[customerEcomm['customer_state'].isin(customerEcomm['customer_state'].value_counts().head(4).index)],
                  col='customer_state', height=3, col_wrap=2)
gvth.map(sns.histplot, 'customer_zip_code_prefix', bins=20)
plt.show()

# Pairplot for Multivariate Analysis (numerical only)
cust_numericaldata = ['customer_zip_code_prefix']
if len(cust_numericaldata) > 1: 
    sns.pairplot(customerEcomm[cust_numericaldata])
    plt.show()
