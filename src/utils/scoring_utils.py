import pandas as pd
import os
import pickle
from joblib import dump, load
from sklearn.metrics import accuracy_score

def Drop_Missing_Val_Columns(df, col_list_to_drop):
    df.drop(columns=col_list_to_drop, inplace=True)
    return df

def Fill_missing_Val_Columns(df, col_list_to_fill):
    for i in col_list_to_fill:
        df[i] = df[i].fillna(df[i].value_counts().index[0])
            #print(df[i].isna().sum())
    return df

def replace_values_using_dict(df, dict_to_replace_values):
    df = df.replace(dict_to_replace_values)
    return df

def split_into_XnY(df, dep_col):
    Ytrain = df[dep_col]
    Xtrain = df.drop([dep_col], axis=1)
    return Xtrain, Ytrain

def predictions(testdf, dep_col, features, model_file_path):
    Xtest, Ytest = split_into_XnY(testdf, dep_col)
    Xtest_tenfeat = Xtest[features]
    mdl = load(model_file_path)
    y_pred = mdl.predict(Xtest_tenfeat)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = ['Y_Predicted']
    merged_df = pd.concat([Xtest,y_pred_df], axis=1)
    return merged_df

