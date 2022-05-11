import pandas as pd
import os
import numpy as np
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

def spearman_corr(df, dep_col):
    spear_coef = df.corr(method="spearman")[dep_col]
    return spear_coef

def convert_result_series_to_df(res, colname_list):
    res_df=pd.DataFrame(res)
    res_df.columns = colname_list
    return res_df

def chisquare_test(df, dep_col):
    list_chi_score = []
    list_col_names = []
    for i in df.columns:
        X = np.array(df[i]).reshape(-1,1)
        y = df[dep_col]
        chi_scores = chi2(X,y)
        list_chi_score.append(chi_scores[1][0])
        chi_score_series = pd.Series( v for v in list_chi_score )
        list_col_names.append(i)
        chi_score_series.index = list_col_names
        chi_score_series.sort_values(ascending=False)
    return chi_score_series

def convert_list_to_series(lst):
    series = pd.Series( v for v in lst )
    return series

def col_list_with_1_class(df, num_list):
    col_list_to_drop = []
    for i in num_list:
        if len(df[i].value_counts()) ==1:
            col_list_to_drop.append(i)
    return col_list_to_drop

def mutualinfo_values(xtrain, ytrain):
    mutual_info = mutual_info_classif(xtrain, ytrain)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = xtrain.columns
    mutual_info.sort_values(ascending=False)
    return mutual_info

def woe_iv_values(xtrain, ytrain):
    woe_iv_sum_list =[]
    col_list= []
    for i in xtrain.columns:
        df_woe_iv = (pd.crosstab(xtrain[i],ytrain,
                        normalize='columns')
                .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
                .assign(iv=lambda dfx: np.sum(dfx['woe']*
                                            (dfx[1]-dfx[0]))))
        #print(df_woe_iv)
        woe_iv_sum_list.append(df_woe_iv.iv.sum())
        woe_iv_sum_series = pd.Series( v for v in woe_iv_sum_list )
        col_list.append(i)
        woe_iv_sum_series.index = col_list
        woe_iv_sum_series.sort_values(ascending=False)
    return woe_iv_sum_series

def Analysis_Report(list_of_df, key):
    merged_df = pd.concat((list_of_df), axis=1)
    return merged_df

def selectkbest_features_mutual_classif(xtrain, ytrain, K):
    sel_five_cols = SelectKBest(mutual_info_classif, k=K)
    sel_five_cols.fit(xtrain, ytrain)
    five_best_features = list(xtrain.columns[sel_five_cols.get_support()])
    return five_best_features

def modelling_data_linReg_n_Feature_selection(xtrain, ytrain, k_features, forward):
    ##Finding Best features using Backward Algorithm with Linear Regression 
    log_reg = LogisticRegression()
    sfs_logReg = sfs(log_reg, k_features=10, forward=forward, verbose=2, scoring='neg_mean_squared_error')
    sfs_logReg = sfs_logReg.fit(xtrain, ytrain)
    feat_names = list(sfs_logReg.k_feature_names_)
    return feat_names

def modelling_data_logReg_n_Feature_selection(xtrain, ytrain, k_features, forward):
    ##Finding Best features using Backward Algorithm with Linear Regression 
    lin_reg = LinearRegression()
    sfs_linReg = sfs(lin_reg, k_features=10, forward=forward, verbose=2, scoring='neg_mean_squared_error')
    sfs_linReg = sfs_linReg.fit(xtrain, ytrain)
    feat_names = list(sfs_linReg.k_feature_names_)
    return feat_names

def splitdf_into_Xtrain_n_Ytrain(df, dep_col):
    Ytrain = df[dep_col]
    Xtrain = df.drop([dep_col], axis=1)
    return Xtrain, Ytrain