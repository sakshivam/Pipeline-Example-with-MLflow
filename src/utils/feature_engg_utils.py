import pandas as pd
import os
import numpy as np

def Categorical_Numerical_Features_split(df):
    categorical_data = df.select_dtypes(exclude=[np.number])
    cat_list = list(categorical_data.columns)
    numeric_data = df.select_dtypes(include=[np.number])
    num_list = list(numeric_data.columns)
    return cat_list, num_list

def plot_stackedgraph_categorical(df, cat_list):
    for i in cat_list:
        df.groupby([i])['Y'].apply(lambda x: x.value_counts() / len(x)).transpose().unstack().plot(kind='bar',stacked = True)

def scatterplot_for_numerical_features(df, num_lst):
    num_list_new = num_lst[0:7]
    for i in num_list_new:
        ax1 =df.plot.scatter(x=i, y='Y',c='DarkBlue')

def replace_values_using_dict(df, dict_to_replace_values):
    df = df.replace(dict_to_replace_values)
    return df