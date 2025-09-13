import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


database_Peoples = pd.read_csv('people-100.csv')
print(database_Peoples.head().to_string())
print("\n")
print(database_Peoples.info())
print("\n")
print(database_Peoples.shape)

#Identify and remove exact/partial duplicate rows
DuplicateRowsum = database_Peoples.duplicated().sum()
print("Number of duplicated rows: ",DuplicateRowsum )
print("The Duplicated Rows are: \n", database_Peoples[database_Peoples.duplicated()])
print("\n")
database_withDuplicatedRowsGone = database_Peoples.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", database_withDuplicatedRowsGone.head().to_string())
print("\n")

PartialCopyRowSum = database_Peoples.duplicated(subset= database_Peoples.columns.difference(['Index']) , keep="first").sum()
print("The No People Database Partial Duplicates Rows are: ", PartialCopyRowSum)
print("\n")

PeopleNoduplicate = database_Peoples.drop_duplicates(subset = database_Peoples.columns.difference(['Index']) , keep = 'first')
print(PeopleNoduplicate.head().to_string())



Cross_Columns_people = ['First Name', 'Last Name', 'Email', 'Job Title']

paritalDuplicateRows = []

for index, row in database_Peoples.iterrows():
    for i in range(len(Cross_Columns_people)):
        for j in range(i+1, len(Cross_Columns_people)):
            datacal1 = str(row[Cross_Columns_people[i]]).lower()
            datacal2 = str(row[Cross_Columns_people[j]]).lower()
            if datacal1 and datacal1 != "nan" and datacal1 in datacal2:
                paritalDuplicateRows.append((index, Cross_Columns_people[i], Cross_Columns_people[j], row[Cross_Columns_people[i]], row[Cross_Columns_people[j]]))

print("Rows with cross-column redundancy:")
for r in paritalDuplicateRows:
    print(r)


print("So we are observing that there is no Cross Column Redundancy in the People Dataset \n")

# Rename columns

print(database_Peoples.info())


database_Peoples = database_Peoples.rename(columns={
    "Index": "PeopleUserIndex",
    "User Id": "PeopleUser_Id",
    "First Name": "PeopleUserFname",
    "Last Name": "PeopleUserLname",
    "Sex": "PeopleUserGender",
    "Email": "PeopleUserEmail",
    "Phone": "PeopleUserPhoneNo",
    "Date of birth": "PeopleUserDob",
    "Job Title": "PeopleUserJobdesc"
})

# Reorder columns
database_Peoples = database_Peoples[[
    "PeopleUserIndex",
    "PeopleUser_Id",
    "PeopleUserFname",
    "PeopleUserLname",
    "PeopleUserGender",
    "PeopleUserEmail",
    "PeopleUserPhoneNo",
    "PeopleUserDob",
    "PeopleUserJobdesc"
]]

print(database_Peoples.head().to_string())

print("Missing Values in each Column: \n", database_Peoples.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (database_Peoples.isnull().mean()*100))

Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': database_Peoples.isnull().any(),
    'Missing_Values_Count': database_Peoples.isnull().sum(),
    'Missing_ValuesIn_Percentage': (database_Peoples.isnull().sum() / len(database_Peoples)) * 100
})

print(Missing_Values_Coloums_Dataframe.to_string())

# Deletion: Drop rows/columns if >70% missing

MissingValuePercentage =  database_Peoples.isna().mean()
Database_After_DropedColoums = database_Peoples.loc[:, MissingValuePercentage <= 0.7]

MissingValuePercentagerow =  database_Peoples.isna().mean(axis=1)
Database_After_DropedRows = database_Peoples.loc[MissingValuePercentagerow <= 0.7, :]

print("The Database After Dropping the Coloums with more than 70% Missing Values are : \n", Database_After_DropedColoums.head().to_string())


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
Database_After_Imputation = Imputation_Of_Values(database_Peoples)
print("The Database After Imputation of Missing Values is : \n", Database_After_Imputation.head().to_string())


#Convert strings to datetime objects (e.g., "2023-12-01" → datetime64)
print(database_Peoples.dtypes)
print("\n")

print("Taking Copy of the dataframe for safety purposes\n")
CopyPeopleDataframe = database_Peoples.copy()

def StringDateTimeFunc(dataframe, dataframecols):
    for col in dataframecols:
        dataframe[col] = pd.to_datetime(dataframe[col], format='%Y-%m-%d', errors='coerce')
    return dataframe

# use it only on specific columns
CopyPeopleDataframe = StringDateTimeFunc(database_Peoples, ["PeopleUserDob"])

print(CopyPeopleDataframe.head().to_string())
print("\n")
print("Generating the Info of the Dataframe After Converting the PeopleUserDob Column to Datetime Format \n")
print(CopyPeopleDataframe.info())
print("\n")
print(database_Peoples["PeopleUserDob"].dtype)


# Fix numerical columns trapped as strings (e.g., "1,000" → 1000)

print("Taking Copy of the dataframe for safety purposes\n")
CopyPeopleDataframe = database_Peoples.copy()

print(database_Peoples.dtypes)
print("\n")
def StringCoverisonFunction(DataFrame, ColoumsDataframe):
    for col in ColoumsDataframe:
        peoplcont = []
        for datapeopleinst in DataFrame[col]:
            datapeopleinst = str(datapeopleinst)
            datapeopleinst = datapeopleinst.replace(".", "-").replace(" ", "-")
            if "x" in datapeopleinst:
                datapeopleinst = datapeopleinst.split("x")[0]
            datapeopleinst = datapeopleinst.strip("-")
            peoplcont.append(datapeopleinst.strip())
        DataFrame[col] = peoplcont
    return DataFrame
DatabasePoplefinal = StringCoverisonFunction(database_Peoples, ['PeopleUserPhoneNo'])

print("The Database After Fixing Numerical Columns is : \n", DatabasePoplefinal.head().to_string())

print("\n")

print("Phone numbers after cleaning:\n", DatabasePoplefinal[["PeopleUserPhoneNo"]].head().to_string())
print(DatabasePoplefinal.dtypes)


#Standardize categorical encoding
from sklearn.preprocessing import LabelEncoder
def CategoricalEncodeFunc(database, database_coloums):
    peopledfcont = {}

    for col in database_coloums:
        codecont = LabelEncoder()
        database[col] = database[col].astype(str).str.strip().str.lower()

        
        database[col + "_code"] = codecont.fit_transform(database[col])
        peopledfcont[col] = codecont

    return database, peopledfcont

print("Assuming Here the Categorical Columns are : columns=   ['PeopleUserGender', 'PeopleUserJobdesc'] \n")
modifieddataframe,peopledfcont = CategoricalEncodeFunc(database_Peoples,   ['PeopleUserGender', 'PeopleUserJobdesc'])
print("The Database After Standardization of Categorical Columns is : \n", modifieddataframe.head().to_string())

def TextNormalizeFunc(database,ordcols):
    for col in ordcols:
        database[col] = database[col].astype(str).str.strip().str.lower()
    return database
print("Assuming Here the Text Columns are : ['PeopleUserFname', 'PeopleUserLname', 'PeopleUserGender', 'PeopleUserEmail', 'PeopleUserJobdesc'] \n")
peopliseframe = TextNormalizeFunc(database_Peoples,['PeopleUserFname', 'PeopleUserLname', 'PeopleUserGender', 'PeopleUserEmail', 'PeopleUserJobdesc'])
print("The Database After Normalization of Text Columns is : \n", peopliseframe.head().to_string())


#- Map categorical variants (e.g.,"M"/"Male" → "Male")
print("Taking Copy of the dataframe for safety purposes\n")
CopyPeopleDataframe = database_Peoples.copy()

def peopleGenderChange(database):
    database["PeopleUserGender"] = database["PeopleUserGender"].astype(str).str.strip().str.lower()
    database["PeopleUserGender"] = database["PeopleUserGender"].replace({
    "m": "Male","male": "Male","f": "Female","female": "Female",
    "others": "Other","o": "Other"
    
})
peopleGenderChange(database_Peoples)
print("In the People Dataset After the Gender Changes Has been Completd:\n", database_Peoples[["PeopleUserGender"]].head(10).to_string())


# Fix inconsistent units (e.g., "kg"vs "lbs")
print("\n")
print(database_Peoples.head().to_string())
print("\n")
print(database_Peoples.info())
print("\n")
print(database_Peoples.shape)
print("\n")
print("The Second of the  Fix inconsistent units (e.g., kg vs lbs)\n" \
"This is Not UseFull at all for the Given Organization Dataset As there are No Attributes Depecting Metric Measurements So Not Implemented For this Particular DatasetCode Alone\n")