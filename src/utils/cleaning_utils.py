import pandas as pd
import os

def Columns_to_drop(df):
    #Find out which columns have null values and total number of Nan values in that column
    # Determine percentage of missing data in a particular column for cleaning of the Data. 
    # Columns in data with more than 95% null should be dropped.
    null_count = df.isna().sum()
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_val_columns =list(percent_missing[percent_missing > 95].index)
    return missing_val_columns

def Missing_Val_Columns_to_fill(df):
    #Finding the value counts for columns with percent_missing between 0 and 95%
    #Replacing Nan values in each of these columns with most frequent value in that column.
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_val_col_list_to_fill = list(percent_missing[(percent_missing > 0) & (percent_missing < 95)].index)
    
    return missing_val_col_list_to_fill

def Drop_Missing_Val_Columns(df, col_list_to_drop):
    df.drop(columns=col_list_to_drop, inplace=True)
    return df

def Fill_missing_Val_Columns(df, col_list_to_fill):
    for i in col_list_to_fill:
        df[i] = df[i].fillna(df[i].value_counts().index[0])
            #print(df[i].isna().sum())
    return df