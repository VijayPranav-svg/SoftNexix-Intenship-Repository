import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from scipy import stats


StockMarketAna = pd.read_csv('all_stocks_5yr.csv')
print(StockMarketAna.head().to_string())
print("\n")
print(StockMarketAna.info())
print("\n")
print(StockMarketAna.shape)


print("Data Shape:", StockMarketAna.shape)
print("\nData Types:\n", StockMarketAna.dtypes)
print("\nMissing Values:\n", StockMarketAna.isnull().sum())
print("\nSummary Statistics of Stock Prices (2013-2018):\n", StockMarketAna.describe(include='all'))



print("\n Handling Missing Values By Filling Them with Median and Mode Values Respectively \n")

StockMarketAna['open']  = StockMarketAna['open'].fillna(StockMarketAna['open'].mean())
StockMarketAna['high']  = StockMarketAna['high'].fillna(StockMarketAna['high'].mean())
StockMarketAna['low']   = StockMarketAna['low'].fillna(StockMarketAna['low'].mean())

StockMarketAna['Name'] = StockMarketAna['Name'].fillna(StockMarketAna['Name'].mode()[0])
print("The Missing Values After Handling is:\n", StockMarketAna.isnull().sum())
print("\nThe Shape After Handling is:\n", StockMarketAna.shape)
print("\n The First 5 Rows After Handling is:\n", StockMarketAna.head().to_string())

# 1. Histogram of Closing Prices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(StockMarketAna['close'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Close Price Distribution')


highstocks = StockMarketAna.groupby('Name')['volume'].mean().nlargest(10).index
hightstockdata = StockMarketAna[StockMarketAna['Name'].isin(highstocks)]

print(" \n I am Using Boxplot for top 10 companies \n")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Name', y='volume', data=hightstockdata)
plt.title('Volume Distribution for Top 10 Companies by Avg Volume')
plt.xticks(rotation=45)
plt.show()


print("\n I am Using Count of Records (Top 10 Companies with most rows) \n ") 
highcount = StockMarketAna['Name'].value_counts().nlargest(10).index
hightcount_Data = StockMarketAna[StockMarketAna['Name'].isin(highcount)]

plt.figure(figsize=(8, 5))
sns.countplot(x='Name', data=hightcount_Data, order=highcount)
plt.title('Number of Records (Top 10 Companies)')
plt.xticks(rotation=45)
plt.show()


print("\n I am Using a Heatmap of Correlations between Numeric Features \n") 
plt.figure(figsize=(10, 6))
sns.heatmap(StockMarketAna.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Stock Features")
plt.show()

StockMarketAna['Close_Category'] = pd.qcut(StockMarketAna['close'], q=3, labels=['Low', 'Medium', 'High'])

crosstab = pd.crosstab(StockMarketAna['Name'], StockMarketAna['Close_Category'], normalize='index') * 100
print("\nPercentage Distribution of Close Price Categories by Company:\n")
print(crosstab)


print(StockMarketAna.head().to_string())


print("I am Using Boxplot to Detect Outliers in Closing Prices \n")
plt.figure(figsize=(8, 4))
sns.boxplot(x=StockMarketAna['close']).set_title('Closing Price Outliers')
z_scores = np.abs(stats.zscore(StockMarketAna['close']))
Stockoutlier = StockMarketAna[z_scores > 3]
print(f"Found {len(Stockoutlier)} closing price outliers")



fivehighstock = StockMarketAna['Name'].value_counts().nlargest(5).index
fivstockdata = StockMarketAna[StockMarketAna['Name'].isin(fivehighstock)]
g = sns.FacetGrid(fivstockdata, col='Close_Category', row='Name', height=2, aspect=1.5)
g.map(sns.histplot, 'close', bins=20)
plt.show()

Teststockdata = StockMarketAna.sample(15000, random_state=42)
sns.pairplot(
    Teststockdata[['open', 'close', 'volume', 'Close_Category']], 
    hue='Close_Category'
)
plt.suptitle("Pairplot of Stock Features by Close Category (Teststockdata)", y=1.02)
plt.show()