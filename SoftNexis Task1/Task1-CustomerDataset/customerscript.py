import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


database = pd.read_csv('customers-100.csv')
print(database.head())
print(database.info())
print(database.shape)

print("Number of duplicated rows: ", database.duplicated().sum())
print("The Duplicated Rows are: \n", database[database.duplicated()])
print("\n")
database_no_duplicated_rows = database.drop_duplicates()


Partial_Duplicates_Rows_Count = database.duplicated(subset= database.columns.difference(['Index']) , keep=False).sum()
print("The No of Partial Duplicates Rows are: ", Partial_Duplicates_Rows_Count)
print("\n")
database_no_partial_duplicates = database.drop_duplicates(subset = database.columns.difference(['Index']) , keep = 'first')
print(database_no_partial_duplicates.to_string())


# Detect cross-column redundant entries
Cross_Column_Reduntant_Entries = database[database.nunique(axis = 1) < database.shape[1]]
print("The Rows With the Cross Column Redundant Entries are : \n " , Cross_Column_Reduntant_Entries)
database_no_cross_column_redundant_entries = database.drop(Cross_Column_Reduntant_Entries.index)


#Assuming the irrelevant columns are 'Index', 'Company', 'Phone 2', and 'Website'
database_relevant_columns_present = database.drop(columns = ['Index', 'Company', 'Phone 2', 'Website'])
print("The Database with Relevant Columns are: \n", database_relevant_columns_present.to_string())
print("\n Also This is Observed Not Need for this Dataset So Only Shown Once Not Implemented In the Maind database Dataframe \n")


# Reorder columns logically
New_Selected_Column_Order_For_The_Dataset = ['Index' ,'Customer Id', 'First Name', 'Last Name', 'Phone 1' , 'Phone 2', 'Email', 'Company', 'Website' , 'Country', 'City', 'Subscription Date', ]
database_reordered_with_new_column_order = database[New_Selected_Column_Order_For_The_Dataset]
print("The Database with the New Column Order is : \n", database_reordered_with_new_column_order.head().to_string())

print(database.info())
print("\n")
print("The Coloums with the Missing Values in them are : \n", database.isnull().any())
print("\n")
print("The No of Missing Values in each Column  and every Column are : \n", database.isnull().sum())


Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': database.isnull().any(),
    'Missing_Values_Count': database.isnull().sum(),
    'Missing_ValuesIn_Percentage': (database.isnull().sum() / len(database)) * 100
})

print(Missing_Values_Coloums_Dataframe.to_string())



#Deletion: Drop rows/columns if >70% missing

MissingValuePercentage =  database.isna().mean()
Database_After_DropedColoums = database.loc[:, MissingValuePercentage <= 0.7]

MissingValuePercentagerow =  database.isna().mean(axis=1)
Database_After_DropedRows = database.loc[MissingValuePercentagerow <= 0.7, :]

print("The Database After Dropping the Coloums with more than 70% Missing Values are : \n", Database_After_DropedColoums.to_string())


# Imputation: Fill with mean/median (numerical), mode (categorical), or interpolate (time-series)
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
Database_After_Imputation = Imputation_Of_Values(database)
print("The Database After Imputation of Missing Values is : \n", Database_After_Imputation.to_string())

#Convert strings to datetime objects (e.g., "2023-12-01" → datetime64)
print(database.dtypes)
print("\n")
def Converting_ToThe_DatetimeFormat(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# use it only on specific columns
Database_After_Conversion = Converting_ToThe_DatetimeFormat(database, ['Subscription Date'])
print("The Database After Converting Strings to Datetime is : \n", Database_After_Conversion.to_string())

print("\n")
print(Database_After_Conversion.head().to_string())


print("From Observation What is Observed for this dataset this is not required as all the columns are in correct format no need to convert them again it only results in eveying converting to NaT or NaN")


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

print("Assuming Here the Categorical Columns are : ['First Name', 'Last Name', 'Email', 'Country', 'City'] \n")
modifieddataframe,customercodeholder = CategoricalEncodeFunc(database, ['First Name', 'Last Name', 'Email', 'Country', 'City'])
print("The Database After Standardization of Categorical Columns is : \n", modifieddataframe.to_string())

    

#Normalize text (lowercase, strip whitespace)

print("The Second  & Third One of the  Fix inconsistent units (e.g., kg vs lbs)- Map categorical variants (e.g., M/Male → Male) \n" \
"This is Not UseFull at all for the Given Customer Dataset As there are No Attributes Depecting gender or Metric Measurements So Not Implemented For this Particular DatasetCode Alone\n")

def TextNormalizeFunc(database,database_coloums):
    for col in database_coloums:
        database[col] = database[col].astype(str).str.strip().str.lower()
    return database
print("Assuming Here the Text Columns are : ['First Name', 'Last Name', 'Email', 'Country', 'City'] \n")
normalizedframe = TextNormalizeFunc(database, ['First Name', 'Last Name', 'Email', 'Country', 'City'])
print("The Database After Normalization of Text Columns is : \n", normalizedframe.to_string())


print("Final Result Obtained After All the Data Cleaning and PreProcessing Steps are : \n")

print(database.info())
print(database.describe())
print(database.head().to_string())



