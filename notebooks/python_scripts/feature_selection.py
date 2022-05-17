#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import os
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# In[22]:


# !pip install mlxtend


# In[23]:


current_path = os.getcwd()
dirname, filename = os.path.split(current_path)
print(dirname, filename)


# In[24]:


data_dir = os.path.join(dirname, 'files')
print(data_dir)


# In[25]:


train_fengg_file_path = os.path.join(data_dir,
                                     "step3\\traindf_with_feature_engg.parquet")
traindf_with_feature_engg = pd.read_parquet(train_fengg_file_path)


# In[26]:


traindf_with_feature_engg.head()


# ## Analysis Functions

# In[27]:


def spearman_corr(df, dep_col):
    spear_coef = df.corr(method="spearman")[dep_col]
    return spear_coef


# In[28]:


def convert_result_series_to_df(res, colname_list):
    res_df = pd.DataFrame(res)
    res_df.columns = colname_list
    return res_df


# In[29]:


def chisquare_test(df, dep_col):
    list_chi_score = []
    list_col_names = []
    for i in df.columns:
        X = np.array(df[i]).reshape(-1, 1)
        y = df[dep_col]
        chi_scores = chi2(X, y)
        list_chi_score.append(chi_scores[1][0])
        chi_score_series = pd.Series(v for v in list_chi_score)
        list_col_names.append(i)
        chi_score_series.index = list_col_names
        chi_score_series.sort_values(ascending=False)
    return chi_score_series


# In[30]:


def convert_list_to_series(lst):
    series = pd.Series(v for v in lst)
    return series


# In[31]:


def col_list_with_1_class(df, num_list):
    col_list_to_drop = []
    for i in num_list:
        if len(df[i].value_counts()) == 1:
            col_list_to_drop.append(i)
    return col_list_to_drop


# In[32]:


def Categorical_Numerical_Features_split(df):
    categorical_data = df.select_dtypes(exclude=[np.number])
    cat_list = list(categorical_data.columns)
    numeric_data = df.select_dtypes(include=[np.number])
    num_list = list(numeric_data.columns)
    return cat_list, num_list


# In[33]:


def mutualinfo_values(xtrain, ytrain):
    mutual_info = mutual_info_classif(xtrain, ytrain)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = xtrain.columns
    mutual_info.sort_values(ascending=False)
    return mutual_info


# In[34]:


def woe_iv_values(xtrain, ytrain):
    woe_iv_sum_list = []
    col_list = []
    for i in xtrain.columns:
        df_woe_iv = (pd.crosstab(xtrain[i], ytrain, normalize='columns')
                     .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
                     .assign(iv=lambda dfx: np.sum(dfx['woe']*(dfx[1]-dfx[0]))
                             ))
        # print(df_woe_iv)
        woe_iv_sum_list.append(df_woe_iv.iv.sum())
        woe_iv_sum_series = pd.Series(v for v in woe_iv_sum_list)
        col_list.append(i)
        woe_iv_sum_series.index = col_list
        woe_iv_sum_series.sort_values(ascending=False)
    return woe_iv_sum_series


# In[35]:


def Analysis_Report(list_of_df, key):
    merged_df = pd.concat((list_of_df), axis=1)
    return merged_df


# In[36]:


def selectkbest_features_mutual_classif(xtrain, ytrain, K):
    sel_five_cols = SelectKBest(mutual_info_classif, k=K)
    sel_five_cols.fit(xtrain, ytrain)
    five_best_features = list(xtrain.columns[sel_five_cols.get_support()])
    return five_best_features

# In[37]:


def modelling_data_linReg_n_Feature_selection(xtrain, ytrain, k_feat, forward):
    # Finding Best features using Backward Algorithm with Linear Regression
    log_reg = LogisticRegression()
    sfs_logReg = sfs(log_reg, k_feat=10, forward=forward,
                     verbose=2, scoring='neg_mean_squared_error')
    sfs_logReg = sfs_logReg.fit(xtrain, ytrain)
    feat_names = list(sfs_logReg.k_feature_names_)
    return feat_names


# In[38]:


def modelling_data_logReg_n_Feature_selection(xtrain, ytrain, k_feat, forward):
    # Finding Best features using Backward Algorithm with Linear Regression
    lin_reg = LinearRegression()
    sfs_linReg = sfs(lin_reg, k_feat=10, forward=forward,
                     verbose=2, scoring='neg_mean_squared_error')
    sfs_linReg = sfs_linReg.fit(xtrain, ytrain)
    feat_names = list(sfs_linReg.k_feature_names_)
    return feat_names


# ## Action Functions

# In[39]:


def splitdf_into_Xtrain_n_Ytrain(df, dep_col):
    Ytrain = df[dep_col]
    Xtrain = df.drop([dep_col], axis=1)
    return Xtrain, Ytrain


# ## Flow

# In[41]:


res_spear_coef = spearman_corr(traindf_with_feature_engg, 'Y')
spear_coef_df = convert_result_series_to_df(res_spear_coef,
                                            colname_list=['spearman_coefficient'])


# In[42]:


lst_score_series = chisquare_test(traindf_with_feature_engg, 'Y')
chi_square_score_df = convert_result_series_to_df(lst_score_series,
                                                  ['chi_square_scores'])


# In[43]:


categorical_list, nume_list = Categorical_Numerical_Features_split(traindf_with_feature_engg)


# In[44]:


DROP_COL_LIST_WITH_ONE_CLASS = col_list_with_1_class(traindf_with_feature_engg, nume_list)
print(DROP_COL_LIST_WITH_ONE_CLASS)


# In[45]:


Xtrain, Ytrain = splitdf_into_Xtrain_n_Ytrain(traindf_with_feature_engg, 'Y')


# In[46]:


mutual_info_series = mutualinfo_values(Xtrain, Ytrain)
mutual_info_df = convert_result_series_to_df(mutual_info_series,
                                             ['mutual_info_values'])


# In[47]:


weight_of_evidence_Info_series = woe_iv_values(Xtrain, Ytrain)
WOE_IV_Sum_df = convert_result_series_to_df(weight_of_evidence_Info_series,
                                            ['WOE_IV_values'])


# In[54]:


dfs_list = [spear_coef_df[:-1], chi_square_score_df[:-1],
            mutual_info_df, WOE_IV_Sum_df]
key = mutual_info_df.index
Analysisdf = Analysis_Report(dfs_list, key)


# In[55]:


train_fsel_analysis_report_file_path = os.path.join(data_dir,
                                                    "step4\\Analysis_report.parquet")
Analysisdf.to_parquet(train_fsel_analysis_report_file_path, index=False)


# In[49]:


TEN_BEST_FEATURES_WITH_MUTUAL_INFO_CLASSIF = selectkbest_features_mutual_classif(Xtrain, Ytrain, 10)
TEN_BEST_FEATURES_WITH_MUTUAL_INFO_CLASSIF


# In[50]:


TEN_BEST_FEATURES_WITH_LIN_REG_BKWD = modelling_data_linReg_n_Feature_selection(Xtrain, Ytrain, 10,  False)
TEN_BEST_FEATURES_WITH_LIN_REG_BKWD


# In[52]:


TEN_BEST_FEATURES_WITH_LIN_REG_FWD = modelling_data_linReg_n_Feature_selection(Xtrain, Ytrain, 10,  True)
TEN_BEST_FEATURES_WITH_LIN_REG_FWD


# In[51]:


TEN_BEST_FEATURES_WITH_LOG_REG_BKWD = modelling_data_logReg_n_Feature_selection(Xtrain, Ytrain, 10,  False)
TEN_BEST_FEATURES_WITH_LOG_REG_BKWD


# In[53]:


TEN_BEST_FEATURES_WITH_LOG_REG_FWD = modelling_data_logReg_n_Feature_selection(Xtrain, Ytrain, 10,  True)
TEN_BEST_FEATURES_WITH_LOG_REG_FWD


# ## Remarks

# ### CONFIG

# In[ ]:


DROP_COL_LIST_WITH_ONE_CLASS = ['toCoupon_GEQ5min']
TEN_BEST_FEATURES_WITH_MUTUAL_INFO_CLASSIF = ['destination', 'passanger',
                                              'weather', 'time', 'coupon',
                                              'expiration', 'has_children',
                                              'income', 'CoffeeHouse',
                                              'direction_same']
TEN_BEST_FEATURES_WITH_LIN_REG_BKWD = ['destination', 'weather', 'coupon',
                                       'expiration', 'gender', 'maritalStatus',
                                       'occupation', 'income',
                                       'CoffeeHouse', 'direction_opp']
TEN_BEST_FEATURES_WITH_LIN_REG_FWD = ['destination', 'passanger', 'time',
                                      'coupon', 'expiration', 'occupation',
                                      'Bar', 'CoffeeHouse', 'Restaurant20To50',
                                      'direction_same']
TEN_BEST_FEATURES_WITH_LOG_REG_BKWD = ['destination', 'passanger', 'weather',
                                       'coupon', 'expiration', 'gender',
                                       'occupation', 'income', 'CoffeeHouse',
                                       'direction_same']
TEN_BEST_FEATURES_WITH_LOG_REG_FWD = ['destination', 'passanger', 'weather',
                                      'coupon', 'expiration', 'gender',
                                      'occupation', 'CoffeeHouse',
                                      'toCoupon_GEQ15min', 'direction_same']


# In[ ]:
