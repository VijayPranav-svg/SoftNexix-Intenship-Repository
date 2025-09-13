import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

database_Orgination = pd.read_csv('organizations-100.csv')
print(database_Orgination.head().to_string())
print("\n")
print(database_Orgination.info())
print("\n")
print(database_Orgination.shape)

#Identify and remove exact/partial duplicate rows
DuplicateRowsum = database_Orgination.duplicated().sum()
print("Number of duplicated rows: ",DuplicateRowsum )
print("The Duplicated Rows are: \n", database_Orgination[database_Orgination.duplicated()])
print("\n")
database_withDuplicatedRowsGone = database_Orgination.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", database_withDuplicatedRowsGone.head().to_string())
print("\n")
PartialCopyRowSum = database_Orgination.duplicated(subset= database_Orgination.columns.difference(['Index']) , keep="first").sum()
print("The No Orinization Database Partial Duplicates Rows are: ", PartialCopyRowSum)
print("\n")

OrginationNoduplicate = database_Orgination.drop_duplicates(subset = database_Orgination.columns.difference(['Index']) , keep = 'first')
print(OrginationNoduplicate.head().to_string())

#  cross-column redundancy
Cross_Coloums_datacheck = ['Organization Id', 'Name', 'Website', 'Country', 'Description', 'Industry']

paritalDuplicateRows = []

for index, row in database_Orgination.iterrows():
    for i in range(len(Cross_Coloums_datacheck)):
        for j in range(i+1, len(Cross_Coloums_datacheck)):
            datacal1 = str(row[Cross_Coloums_datacheck[i]]).lower()
            datacal2 = str(row[Cross_Coloums_datacheck[j]]).lower()
            if datacal1 and datacal1 != "nan" and datacal1 in datacal2:
                paritalDuplicateRows.append((index, Cross_Coloums_datacheck[i], Cross_Coloums_datacheck[j], row[Cross_Coloums_datacheck[i]], row[Cross_Coloums_datacheck[j]]))

print("Rows with cross-column redundancy:")
for r in paritalDuplicateRows:
    print(r)

print("So we are observing that there is no Cross Column Redundancy in the Orginazation Dataset \n")


# Rename columns
database_Orgination = database_Orgination.rename(columns={
    "Index": "Orginazationindex",
    "Organization Id": "organization_id",
    "Name": "Organizationname",
    "Website": "Databasewebsite",
    "Country": "Country",
    "Description": "Countrydescription",
    "Founded": "founded_year",
    "Industry": "industry",
    "Number of employees": "num_employees"
})

# Reorder columns
database_Orgination = database_Orgination[[
    "Orginazationindex", "organization_id", "Organizationname", "Databasewebsite",
    "industry", "founded_year", "num_employees",
    "Country", "Countrydescription"
]]

print(database_Orgination.head().to_string())



print("Missing Values in each Column: \n", database_Orgination.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (database_Orgination.isnull().mean()*100))

Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': database_Orgination.isnull().any(),
    'Missing_Values_Count': database_Orgination.isnull().sum(),
    'Missing_ValuesIn_Percentage': (database_Orgination.isnull().sum() / len(database_Orgination)) * 100
})

print(Missing_Values_Coloums_Dataframe.to_string())

# Deletion: Drop rows/columns if >70% missing

MissingValuePercentage =  database_Orgination.isna().mean()
Database_After_DropedColoums = database_Orgination.loc[:, MissingValuePercentage <= 0.7]

MissingValuePercentagerow =  database_Orgination.isna().mean(axis=1)
Database_After_DropedRows = database_Orgination.loc[MissingValuePercentagerow <= 0.7, :]

print("The Database After Dropping the Coloums with more than 70% Missing Values are : \n", Database_After_DropedColoums.to_string())


def Imputation_Of_Values(database):
    for column in database.columns:
        if np.issubdtype(database[column].dtype, np.number):
            median_value = database[column].median()
            database[column].fillna(median_value , inplace=True)
            print(f"Filled NaN in numeric column '{column}' with mean = {median_value}")
        
        elif database[column].dtype == 'object':
            mode_value = database[column].mode()[0]
            database[column].fillna(mode_value , inplace=True)
            print(f"Filled NaN in categorical column '{column}' with mode = {mode_value}")
        
        elif np.issubdtype(database[column].dtype, np.datetime64):
            database[column] = database[column].interpolate(method='time')
            print(f"Interpolated missing datetime values in '{column}'")
    
    return database
Database_After_Imputation = Imputation_Of_Values(database_Orgination)
print("The Database After Imputation of Missing Values is : \n", Database_After_Imputation.head().to_string())

print("First Creating a Copy of the Datframe and then  working on the Following steps \n ")

CopyOrginazationDataframe = database_Orgination.copy()


#Convert strings to datetime objects (e.g., "2023-12-01" → datetime64)
print(database_Orgination.dtypes)
print("\n")

def StringDateTimeFunc(dataframe, dataframecols):
    for col in dataframecols:
        dataframe[col] = pd.to_datetime(dataframe[col], format='%Y', errors='coerce')
    return dataframe

# use it only on specific columns
ConvertedOrginazationDataframe = StringDateTimeFunc(database_Orgination, ['founded_year'])

print(ConvertedOrginazationDataframe.head().to_string())
print("\n")
print("Generating the Info of the Dataframe After Converting the Founded Year Column to Datetime Format \n")
print(ConvertedOrginazationDataframe.info())
print("\n")
print(database_Orgination['founded_year'].dtype)

# Fix numerical columns trapped as strings (e.g., "1,000" → 1000)

print("Taking Copy of the dataframe for safety purposes\n")
CopyOrginazationDataframe = database_Orgination.copy()

print(database_Orgination.dtypes)
print("\n")
def StringCoverisonFunction(DataFrame, ColoumsDataframe):
    for col in ColoumsDataframe:
        DataFrame[col] = (
                  DataFrame[col]
                  .astype(str)                        
                  .str.replace(',', '', regex=False)  
                  .str.strip()                        
        )
        DataFrame[col] = pd.to_numeric(DataFrame[col], errors='coerce').astype('Int64')
        
    return DataFrame
DatabaseTransformed = StringCoverisonFunction(database_Orgination, ['num_employees'])

print("The Database After Fixing Numerical Columns is : \n", DatabaseTransformed.head().to_string())
#Standardize categorical encoding
from sklearn.preprocessing import LabelEncoder
def CategoricalEncodeFunc(database, database_coloums):
    customercodeholder = {}

    for col in database_coloums:
        codecont = LabelEncoder()
        database[col] = database[col].astype(str).str.strip().str.lower()

        
        database[col + "_code"] = codecont.fit_transform(database[col])
        customercodeholder[col] = codecont

    return database, customercodeholder

print("Assuming Here the Categorical Columns are : columns= ['industry', 'Country'] \n")
modifieddataframe,customercodeholder = CategoricalEncodeFunc(database_Orgination, ['industry', 'Country'])
print("The Database After Standardization of Categorical Columns is : \n", modifieddataframe.head().to_string())

# Normalize text (lowercase, strip whitespace)


def TextNormalizeFunc(database,ordcols):
    for col in ordcols:
        database[col] = database[col].astype(str).str.strip().str.lower()
    return database
print("Assuming Here the Text Columns are : ['Organizationname', 'Databasewebsite', 'industry', 'Country', 'Countrydescription'] \n")
normalizedframe = TextNormalizeFunc(database_Orgination,['Organizationname', 'Databasewebsite', 'industry', 'Country', 'Countrydescription'])
print("The Database After Normalization of Text Columns is : \n", normalizedframe.head().to_string())


# Fix inconsistent units (e.g., "kg" vs "lbs")

#Map categorical variants (e.g., "M"/"Male" → "Male")

print("\n")
print(database_Orgination.head().to_string())
print("\n")
print(database_Orgination.info())
print("\n")
print(database_Orgination.shape)
print("\n")
print("The Second  & Third One of the  Fix inconsistent units (e.g., kg vs lbs)- Map categorical variants (e.g., M/Male → Male) \n" \
"This is Not UseFull at all for the Given Organization Dataset As there are No Attributes Depecting gender or Metric Measurements So Not Implemented For this Particular DatasetCode Alone\n")