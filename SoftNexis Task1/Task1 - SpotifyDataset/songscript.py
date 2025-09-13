import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

spotify_data = pd.read_csv('./Spotify+Streaming+History/spotify_history.csv')
print(spotify_data.head().to_string())
print("\n")
print(spotify_data.info())
print("\n")
print(spotify_data.shape)

DuplicateRowsum = spotify_data.duplicated().sum()
print("Number of duplicated rows: ",DuplicateRowsum )
print("The Duplicated Rows are: \n", spotify_data[spotify_data.duplicated()])
print("\n")
database_withDuplicatedRowsGone = spotify_data.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", database_withDuplicatedRowsGone.head().to_string())
print("\n")


PartialCopyRowSum = spotify_data.duplicated(subset= spotify_data.columns.difference(['ts']) , keep="first").sum()
print("The No Database's  Partial Duplicates Rows are: ", PartialCopyRowSum)
print("\n")

SpotifyNodupe = spotify_data.drop_duplicates(subset = spotify_data.columns.difference(['ts']) , keep = 'first')
print(SpotifyNodupe.head().to_string())


Cross_Columns_spotify = ['track_name', 'artist_name', 'album_name']

paritalDuplicateRows = []

for index, row in spotify_data.iterrows():
    for i in range(len(Cross_Columns_spotify)):
        for j in range(i+1, len(Cross_Columns_spotify)):
            datacal1 = str(row[Cross_Columns_spotify[i]]).lower()
            datacal2 = str(row[Cross_Columns_spotify[j]]).lower()
            if datacal1 and datacal1 != "nan" and datacal1 in datacal2:
                paritalDuplicateRows.append((
                    index,
                    Cross_Columns_spotify[i],
                    Cross_Columns_spotify[j], row[Cross_Columns_spotify[i]],row[Cross_Columns_spotify[j]]
                ))

print("Rows with cross-column redundancy:")
for r in paritalDuplicateRows[:20]:  
    print(r)

if not paritalDuplicateRows:
    print("So We are Observing That there are No cross-column redundancy found in the Spotify dataset.\n")


# Rename columns

print(spotify_data.info())


New_Spotify_Instruction = {

 "spotify_track_uri": "Spoftify_track_uri",
    "ts": "Spotify_TimeStamp",               
    "platform": "StreamingPlatform",          
    "ms_played": "Play_Duration_ms", 
    "track_name": "Track's_Name",
    "artist_name": "Artist'sNames",
    "album_name": "Album'sTitle",
    "reason_start": "Start_reason",
    "reason_end": "End_reason",
    "shuffle": "Shuffle_On",        
    "skipped": "SkippedOn"  


}


spotify_data = spotify_data.rename(columns=New_Spotify_Instruction)


newspotyorder = [
    "Track's_Name",
    "Artist'sNames",
     "Album'sTitle",
    "Start_reason",
    "End_reason",
    "Spoftify_track_uri",
     "Spotify_TimeStamp",               
     "StreamingPlatform",          
     "Play_Duration_ms", 
     "Shuffle_On",        
     "SkippedOn"  
]

# Reorder columns
spotify_data = spotify_data[newspotyorder]
print(spotify_data.head().to_string())



print(spotify_data.info())
print(spotify_data.head().to_string())
print("\n")
print("Missing Values in each Column: \n", spotify_data.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (spotify_data.isnull().mean()*100))

Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': spotify_data.isnull().any(),
    'Missing_Values_Count': spotify_data.isnull().sum(),
    'Missing_ValuesIn_Percentage': (spotify_data.isnull().sum() / len(spotify_data)) * 100
})

print(Missing_Values_Coloums_Dataframe.to_string())


# Deletion: Drop rows/columns if >70% missing

MissingValuePercentage =  spotify_data.isna().mean()
Database_After_DropedColoums = spotify_data.loc[:, MissingValuePercentage <= 0.7]

MissingValuePercentagerow =  spotify_data.isna().mean(axis=1)
Database_After_DropedRows = spotify_data.loc[MissingValuePercentagerow <= 0.7, :]

print("The Database After Dropping the Coloums with more than 70% Missing Values are : \n", Database_After_DropedColoums.head().to_string())


def SpotifyImputation(database):
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
SongDataImputed = SpotifyImputation(spotify_data)
print("The Database After Imputation of Missing Values is : \n", SongDataImputed.head().to_string())


#Convert strings to datetime objects (e.g., "2023-12-01" → datetime64)
print(spotify_data.dtypes)
print("\n")

print("Taking Copy of the dataframe for safety purposes\n")
CopyPeopleDataframe = spotify_data.copy()

def StringDateTimeFunc(dataframe, dataframecols):
    for col in dataframecols:
        dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')
    return dataframe

# use it only on specific columns
CopyPeopleDataframe = StringDateTimeFunc(spotify_data, ["Spotify_TimeStamp"])

print(CopyPeopleDataframe.head().to_string())
print("\n")
print("Generating the Info of the Dataframe After Converting the Spotify_TimeStamp Column to Datetime Format \n")
print(CopyPeopleDataframe.info())
print("\n")
print(spotify_data["Spotify_TimeStamp"].dtype)


# Fix numerical columns trapped as strings (e.g., "1,000" → 1000)

print("Taking Copy of the dataframe for safety purposes\n")
CopyPeopleDataframe = spotify_data.copy()
print("Column datatypes before numeric conversion:\n")
print(spotify_data.dtypes)
print("\n")

print(
    "Result: No action needed.\n"
    "Reason → All numeric columns (like 'Play_Duration_ms') are already stored as proper numeric dtypes.\n"
    "Other object columns (e.g., 'Artist'sNames', 'Album'sTitle', 'Start_reason', 'End_reason') "
    "are categorical text fields, not numbers stored as strings. \n"
    "Therefore, applying numeric string conversion would corrupt these columns, "
    "so the step is skipped.\n"
)
print("The Database After Fixing Numerical Columns is : \n", spotify_data.head().to_string())
print("\n")
#Standardize categorical encoding
categoricalspotify_cols = ["StreamingPlatform", "Start_reason", "End_reason"]
from sklearn.preprocessing import LabelEncoder
def CategoricalEncodeFunc(database, spotifycols):
    spotifysongcont = {}
    for col in spotifycols:
        codecont = LabelEncoder()
        database[col] = database[col].astype(str).str.strip().str.lower()

        
        database[col + "_code"] = codecont.fit_transform(database[col])
        spotifysongcont[col] = codecont

    return database, spotifysongcont

print("Assuming Here the Categorical Columns are : ['StreamingPlatform', 'Start_reason', 'End_reason']\n")
modifieddataframe,spotifysongcont = CategoricalEncodeFunc(spotify_data, categoricalspotify_cols)
print("The Database After Standardization of Categorical Columns is : \n", modifieddataframe.head().to_string())

# Normalize text (lowercase, strip whitespace)
def TextNormalizeFunc(database,ordcols):
    for col in ordcols:
        database[col] = database[col].astype(str).str.strip().str.lower()
    return database

songcols = ["Track's_Name", "Artist'sNames", "Album'sTitle",
                "StreamingPlatform", "Start_reason", "End_reason"]

print(f"Assuming Here the Text Columns are : {songcols}\n")
normalizedframe = TextNormalizeFunc(spotify_data,[])
print("The Database After Normalization of Text Columns is : \n", normalizedframe.head().to_string())

# Fix inconsistent units (e.g., "kg" vs "lbs")
#Map categorical variants (e.g., "M"/"Male" → "Male")

print("\n")
print(spotify_data.head().to_string())
print("\n")
print(spotify_data.info())
print("\n")
print(spotify_data.shape)
print("\n")
print("The Second  & Third One of the  Fix inconsistent units (e.g., kg vs lbs)- Map categorical variants (e.g., M/Male → Male) \n" \
"This is Not UseFull at all for the Given Spotify Dataset As there are No Attributes Depecting gender or Metric Measurements So Not Implemented For this Particular DatasetCode Alone\n")