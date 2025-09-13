
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


print("As the Job Descripting Coloum Data is a Little Large If we Convert Things in To String Not getting proper output so keep it as pure Dataframe itself" \
"")
GlassDoorDataset = pd.read_csv('./Datasets/Uncleaned_DS_jobs.csv')
print(GlassDoorDataset.head())
print("\n")
print(GlassDoorDataset.info())
print("\n")
print(GlassDoorDataset.shape)

DuplicateRowsum = GlassDoorDataset.duplicated().sum()
print("Number of duplicated rows: ",DuplicateRowsum )
print("The Duplicated Rows are: \n", GlassDoorDataset[GlassDoorDataset.duplicated()])
print("\n")
database_withDuplicatedRowsGone = GlassDoorDataset.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", database_withDuplicatedRowsGone.head())
print("\n")

Partial_Duplicates_Rows_Count = GlassDoorDataset.duplicated(subset= GlassDoorDataset.columns.difference(['Index']) , keep=False).sum()
print("The No of Partial Duplicates Rows are: ", Partial_Duplicates_Rows_Count)
print("\n")
GlassDoorDataset_no_partial_duplicates = GlassDoorDataset.drop_duplicates(subset = GlassDoorDataset.columns.difference(['Index']) , keep = 'first')
print(GlassDoorDataset_no_partial_duplicates.head())


glassdorcols = ['Job Title', 'Salary Estimate', 'Job Description', 'Company Name', 
                      'Location', 'Headquarters', 'Industry', 'Sector']

paritalDuplicateRows = []

for index, row in GlassDoorDataset.iterrows():
    for i in range(len(glassdorcols)):
        for j in range(i+1, len(glassdorcols)):
            datacal1 = str(row[glassdorcols[i]]).lower()
            datacal2 = str(row[glassdorcols[j]]).lower()
            if datacal1 and datacal1 != "nan" and datacal1 in datacal2:
                paritalDuplicateRows.append((index, glassdorcols[i], glassdorcols[j], row[glassdorcols[i]], row[glassdorcols[j]]))

print("Rows with cross-column redundancy:")
for r in paritalDuplicateRows:
    print(r)


print(GlassDoorDataset.info())
GlassDoorDataset = GlassDoorDataset.rename(columns={
    "index": "GlassJobIndex",
    "Job Title": "GlassJobTitle",
    "Salary Estimate": "GlassJobSalaryEstimate",
    "Job Description": "GlassJobDescription",
    "Rating": "GlassJobRating",
    "Company Name": "Glass-CompanyName",
    "Location": "GlassJobLocation",
    "Headquarters": "Company-HQ",
    "Size": "CompanySize",
    "Founded": "CompanyFoundedIn",
    "Type of ownership": "CompanyOwnershipType",
    "Industry": "CompanyIndustry",
    "Sector": "CompanySector",
    "Revenue": "CompanyRevenue",
    "Competitors": "CompanyCompetitors"
})


GlassDoorDataset = GlassDoorDataset[[  
    "GlassJobIndex",
    "GlassJobTitle",
    "GlassJobSalaryEstimate",
    "GlassJobDescription",
    "GlassJobRating",
    "Glass-CompanyName",
    "GlassJobLocation",
    "Company-HQ",
    "CompanySize",
    "CompanyFoundedIn",
    "CompanyOwnershipType",
    "CompanyIndustry",
    "CompanySector",
    "CompanyRevenue",
    "CompanyCompetitors"
]]
print("\n")
print(GlassDoorDataset.head())
print("\n")

print("Missing Values in each Column: \n", GlassDoorDataset.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (GlassDoorDataset.isnull().mean()*100))

Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': GlassDoorDataset.isnull().any(),
    'Missing_Values_Count': GlassDoorDataset.isnull().sum(),
    'Missing_ValuesIn_Percentage': (GlassDoorDataset.isnull().sum() / len(GlassDoorDataset)) * 100
})

print(Missing_Values_Coloums_Dataframe.to_string())

# Deletion: Drop rows/columns if >70% missing
MissingValuePercentage =  GlassDoorDataset.isna().mean()
Database_After_DropedColoums = GlassDoorDataset.loc[:, MissingValuePercentage <= 0.7]

MissingValuePercentagerow =  GlassDoorDataset.isna().mean(axis=1)
Database_After_DropedRows = GlassDoorDataset.loc[MissingValuePercentagerow <= 0.7, :]

print("The Database After Dropping the Coloums with more than 70% Missing Values are : \n", Database_After_DropedColoums.head())


def Imputation_Of_Values(database):
    for column in database.columns:
        if np.issubdtype(database[column].dtype, np.number):
            median_value = database[column].median()
            database[column].fillna(median_value , inplace=True)
            print(f"Filled NaN in numeric column '{column}' with Median = {median_value}")
        
        elif database[column].dtype == 'object':
            mode_value = database[column].mode()[0]
            database[column].fillna(mode_value , inplace=True)
            print(f"Filled NaN in categorical column '{column}' with Mode = {mode_value}")
        
        elif np.issubdtype(database[column].dtype, np.datetime64):
            database[column] = database[column].interpolate(method='time')
            print(f"Interpolated missing datetime values in '{column}'")
    
    return database
Database_After_Imputation = Imputation_Of_Values(GlassDoorDataset)
print("The Database After Imputation of Missing Values is : \n", Database_After_Imputation.head())


#Convert strings to datetime objects (e.g., "2023-12-01" → datetime64)
print(GlassDoorDataset.dtypes)
print("\n")

print("Taking Copy of the dataframe for safety purposes\n")
copyglassdoor = GlassDoorDataset.copy()

def StringDateTimeFunc(dataframe, dataframecols):
    for col in dataframecols:
        dataframe[col] = pd.to_datetime(dataframe[col], format='%Y', errors='coerce')
    return dataframe

# use it only on specific columns
copyglassdoor = StringDateTimeFunc(GlassDoorDataset, ["CompanyFoundedIn"])

print(copyglassdoor.head())
print("\n")
print("Generating the Info of the Dataframe After Converting the PeopleUserDob Column to Datetime Format \n")
print(copyglassdoor.info())
print("\n")
print(GlassDoorDataset["CompanyFoundedIn"].dtype)

#Standardize categorical encoding
GlassEncodableCols = [
    'GlassJobTitle',
    'GlassJobLocation',
    'Company-HQ',
    'CompanySize',
    'CompanyOwnershipType',
    'CompanyIndustry',
    'CompanySector',
    'CompanyRevenue'
]
from sklearn.preprocessing import LabelEncoder
def CategoricalEncodeFunc(database, database_coloums):
    gladdrr = {}

    for col in database_coloums:
        codecont = LabelEncoder()
        database[col] = database[col].astype(str).str.strip().str.lower()

        
        database[col + "_code"] = codecont.fit_transform(database[col])
        gladdrr[col] = codecont

    return database, gladdrr

print("Assuming Here the Categorical Columns are : columns=  \n", GlassEncodableCols)
modifieddataframe,gladdrr = CategoricalEncodeFunc(GlassDoorDataset,   GlassEncodableCols)
print("\n")
print("The Database After Standardization of Categorical Columns is : \n", modifieddataframe.head())


def TextNormalizeFunc(database,ordcols):
    for col in ordcols:
        database[col] = database[col].astype(str).str.strip().str.lower()
    return database

GlassDrNormailze = [
    'GlassJobTitle','GlassJobLocation','Company-HQ',
    'CompanySize','CompanyOwnershipType','CompanyIndustry','CompanySector','CompanyRevenue'
]

print("Assuming Here the Text Columns are :  \n", GlassDrNormailze)
glassiseFrame = TextNormalizeFunc(GlassDoorDataset,GlassDrNormailze)
print("\n")
print("The Database After Normalization of Text Columns is : \n", glassiseFrame.head())

# Fix inconsistent units (e.g., "kg" vs "lbs")
#Map categorical variants (e.g., "M"/"Male" → "Male")

print("\n")
print(GlassDoorDataset.head())
print("\n")
print(GlassDoorDataset.info())
print("\n")
print(GlassDoorDataset.shape)
print("\n")
print("The Second  & Third One of the  Fix inconsistent units (e.g., kg vs lbs)- Map categorical variants (e.g., M/Male → Male) \n" \
"This is Not UseFull at all for the Given Organization Dataset As there are No Attributes Depecting gender or Metric Measurements So Not Implemented For this Particular DatasetCode Alone\n")