import pandas as pd
import os
import sys
import yaml
sys.path.insert(0, os.path.abspath('./.'))
sys.path.append(os.path.abspath('././src'))
print("sys.path is", sys.path)
print(os.getcwd())
from src.utils.cleaning_utils import (Drop_Missing_Val_Columns,
                                      Fill_missing_Val_Columns)
from src.utils.feature_engg_utils import replace_values_using_dict

with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

df = pd.read_parquet('tests\\testing_data\\testing_df.parquet',
                     engine='pyarrow')
cleaned_df = pd.read_parquet('tests\\testing_data\\cleaned_df.parquet',
                             engine='pyarrow')


def test_Drop_Missing_Val_Columns():
    global df
    columns_to_drop = config['MISSING_VAL_COLUMNS']
    df = Drop_Missing_Val_Columns(df, columns_to_drop)
    li = list(set(columns_to_drop)-set(df.columns))
    assert li == columns_to_drop


def test_Fill_missing_Val_Columns():
    global df
    columns_to_fill = config['MISSING_VAL_COLUMNS_TO_FILL']
    df = Fill_missing_Val_Columns(df, columns_to_fill)
    for col in columns_to_fill:
        assert df[col].isna().sum() == 0


def test_replace_values_using_dict():
    global cleaned_df
    dict_for_clubbing = config['DICT_FOR_CLUBBING']
    dict_to_get_ordinal_features = config['DICT_TO_GET_ORDINAL_FEATURES']
    cleaned_df = replace_values_using_dict(cleaned_df, dict_for_clubbing)
    cleaned_df = replace_values_using_dict(cleaned_df,
                                           dict_to_get_ordinal_features)
    for col in dict_to_get_ordinal_features.keys():
        setA = set(list(dict_to_get_ordinal_features[col].values()))
        setB = set(cleaned_df[list(dict_to_get_ordinal_features.keys())][col].unique())
    assert setA - setB == set()
