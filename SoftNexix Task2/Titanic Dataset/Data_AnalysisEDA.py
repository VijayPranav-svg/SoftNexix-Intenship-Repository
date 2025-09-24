import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from scipy import stats


titanicdata = pd.read_csv('titanic.csv')
print(titanicdata.head().to_string())
print("\n")
print(titanicdata.info())
print("\n")
print(titanicdata.shape)

print("Data Shape:", titanicdata.shape)
print("\nData Types:\n", titanicdata.dtypes)
print("\nMissing Values:\n", titanicdata.isnull().sum())
print("\nSummary Statistics Of Data:\n", titanicdata.describe(include='all'))
print("Handling Missing Values By Filling Them with Median")
titanicdata.fillna({'Age': titanicdata['Age'].median()}, inplace=True)

fig, titaxix = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(titanicdata['Age'], bins=30, kde=True, ax=titaxix[0]).set_title('Age Distribution')
sns.boxplot(x='Pclass', y='Fare', data=titanicdata, ax=titaxix[1]).set_title('Fare by Class')
plt.figure(figsize=(8, 4))
sns.countplot(x='Sex', hue='Survived', data=titanicdata).set_title('Survival by Gender')


plt.figure(figsize=(10, 6))
sns.heatmap(titanicdata.corr(numeric_only=True), annot=True, cmap='coolwarm')
print(pd.crosstab(titanicdata['Pclass'], titanicdata['Survived'], normalize='index') * 100)
print(titanicdata.head().to_string())


plt.figure(figsize=(8, 4))
sns.boxplot(x=titanicdata['Fare']).set_title('Fare Outliers')
from scipy import stats
z_scores = np.abs(stats.zscore(titanicdata['Fare']))
TitanicOutliers = titanicdata[z_scores > 3]
print(f"Found {len(TitanicOutliers)} fare Outliers")

gt = sns.FacetGrid(titanicdata, col='Survived', row='Pclass', height=3)
gt.map(sns.histplot, 'Age', bins=20)
sns.pairplot(titanicdata[['Age', 'Fare', 'Parents/Children Aboard', 'Survived']], hue='Survived')
