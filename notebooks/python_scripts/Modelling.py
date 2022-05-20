#!/usr/bin/env python
# coding: utf-8

# ## CONFIG

# In[ ]:


import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import lightgbm as lgbm
from sklearn.model_selection import cross_val_score
from joblib import dump

MISSING_VAL_COLUMNS = ['car']
MISSING_VAL_COLUMNS_TO_FILL = ['Bar', 'CoffeeHouse', 'CarryAway',
                               'RestaurantLessThan20', 'Restaurant20To50']

DROP_COL_LIST_WITH_ONE_CLASS = ['toCoupon_GEQ5min']
TEN_BEST_FEATURES_WITH_MUTUAL_INFO_CLASSIF = ['destination', 'passanger',
                                              'weather', 'time', 'coupon',
                                              'expiration', 'has_children',
                                              'income', 'CoffeeHouse',
                                              'direction_same']
TEN_BEST_FEATURES_WITH_LIN_REG_BKWD = ['destination', 'weather', 'coupon',
                                       'expiration', 'gender', 'maritalStatus',
                                       'occupation', 'income', 'CoffeeHouse',
                                       'direction_opp']
TEN_BEST_FEATURES_WITH_LIN_REG_FWD = ['destination', 'passanger', 'time',
                                      'coupon', 'expiration', 'occupation',
                                      'Bar', 'CoffeeHouse', 'Restaurant20To50',
                                      'direction_same']
TEN_BEST_FEATURES_WITH_LOG_REG_BKWD = ['destination', 'passanger', 'weather',
                                       'coupon', 'expiration', 'gender',
                                       'occupation', 'income',
                                       'CoffeeHouse', 'direction_same']
TEN_BEST_FEATURES_WITH_LOG_REG_FWD = ['destination', 'passanger', 'weather',
                                      'coupon', 'expiration', 'gender',
                                      'occupation', 'CoffeeHouse',
                                      'toCoupon_GEQ15min', 'direction_same']
TEN_BEST_FEATURES_OBSERVED_SELECTION = ['passanger', 'coupon', 'CoffeeHouse',
                                        'destination', 'expiration',
                                        'toCoupon_GEQ25min', 'Bar', 'gender',
                                        'Restaurant20To50', 'temperature']


# In[2]:


Categorical_Features = ['destination', 'passanger', 'weather', 'time',
                        'coupon', 'expiration', 'gender', 'age',
                        'maritalStatus', 'education', 'occupation',
                        'income', 'Bar', 'CoffeeHouse', 'CarryAway',
                        'RestaurantLessThan20',
                        'Restaurant20To50']
Numerical_Features = ['temperature', 'has_children', 'toCoupon_GEQ5min',
                      'toCoupon_GEQ15min', 'toCoupon_GEQ25min',
                      'direction_same', 'direction_opp', 'Y']


# In[4]:


# pip install imbalanced-learn==0.7


# In[26]:


# pip install lightgbm


# In[87]:


# In[7]:


current_path = os.getcwd()
dirname, filename = os.path.split(current_path)
print(dirname, filename)


# In[8]:


data_dir = os.path.join(dirname, 'files')
print(data_dir)


# In[9]:


train_fengg_file_path = os.path.join(data_dir,
                                     "step3\\traindf_with_feature_engg.parquet")
traindf_with_feature_engg = pd.read_parquet(train_fengg_file_path)


# In[10]:


traindf_with_feature_engg.head()


# In[11]:


traindf_with_feature_engg.drop(DROP_COL_LIST_WITH_ONE_CLASS,
                               axis=1, inplace=True)


# ## Analysis Functions

# In[ ]:


# Action Functions

# In[12]:


def test_train_split(df, frac_train):
    frac = frac_train
    train_df = df.sample(frac=frac)
    test_df = df.drop(train_df.index)
    return train_df, test_df


# In[13]:


def split_into_XnY(df, dep_col):
    Ytrain = df[dep_col]
    Xtrain = df.drop([dep_col], axis=1)
    return Xtrain, Ytrain


# In[14]:


def statsmodel_logit(xtrain, ytrain, xvalidation, yvalidation):
    logit_model = sm.Logit(ytrain, xtrain).fit()
    print(logit_model.summary2())
    y_pred = logit_model.predict(xvalidation)
    prediction = list(map(round, y_pred))
    print(accuracy_score(yvalidation, prediction))
    return logit_model


# In[15]:


def model_logit_sklearn(xtrain, ytrain, xvalidation, yvalidation):
    logreg = LogisticRegression().fit(xtrain, ytrain)
    y_pred = logreg.predict(xvalidation)
    print(accuracy_score(yvalidation, y_pred))
    return logreg


# In[6]:


def balancing_data_with_SMOTE(xtrain, ytrain):
    # if the dataset is unbalanced i.e no. of values
    # in each level of target variable varies a lot.
    # Then it should be balanced.
    # For this SMOTE is used.
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(xtrain, ytrain)
    print(y_train_sm.value_counts())
    return X_train_sm, y_train_sm


# In[28]:


def GridSearchCV_forBestParams(xtrain, ytrain):
    # Input Xtrain and Ytrain should be balanced
    gbm1 = lgbm.LGBMClassifier({'early_stopping_rounds': 100,
                                'bagging_fraction': 0.5,
                                'bagging_freq': 5,
                                'objective': 'binary',
                                'num_boost_round': 1000,
                                'num_threads': 4,
                                })
    gridParams = {'boosting_type': ['dart', 'gbdt'],
                  'metric': ['binary_logloss', 'binary_error',
                             'auc'],
                  'num_leaves': np.array([8, 16, 32]),
                  'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
                  'max_depth': [2, 3, 4, 5]
                  }
    grid = GridSearchCV(gbm1, gridParams, verbose=2, cv=4, n_jobs=1)
    grid.fit(xtrain, ytrain)
    best_params = grid.best_params_
    return best_params


# In[90]:


def lgbm_model(xtrain, ytrain, xvalidation, yvalidation):
    # Using LGBM Classifier with best parameters and then
    # training the training data
    # Input Xtrain and Ytrain should be balanced
    grid_params = GridSearchCV_forBestParams(xtrain, ytrain)
    parameters = {'objective': 'binary',  'num_boost_round': 1000,
                  'num_threads': 4}
    parameters.update(grid_params)
#     param_str = str(parameters)
#     param_str =param_str.replace(':','=')
#     param_str = param_str.replace('{', '')
#     param_str = param_str.replace('}', '')
    gbm = lgbm.LGBMClassifier(**parameters)
    gbm.fit(xtrain, ytrain, early_stopping_rounds=100,
            eval_set=[(xtrain, ytrain), (xvalidation, yvalidation)])
    # y_pred  = gbm.predict(xvalidation)
    # y_pred_prob = gbm.predict_proba(xvalidation)
    print('Accuracy for Train Set', cross_val_score(gbm, xtrain,
                                                    ytrain, scoring='accuracy',
                                                    cv=4, n_jobs=4))
    print('Accuracy for Validation Set', cross_val_score(gbm, xvalidation,
                                                         yvalidation,
                                                         scoring='accuracy',
                                                         cv=4, n_jobs=4))
    return gbm


# Flow

# In[16]:


fraction_train = 0.7
traindf, validationdf = test_train_split(traindf_with_feature_engg,
                                         fraction_train)


# In[17]:


Xtrain, Ytrain = split_into_XnY(traindf, 'Y')


# In[18]:


Xval, Yval = split_into_XnY(validationdf, 'Y')


# In[19]:


Xtrain_tenfeat = Xtrain[TEN_BEST_FEATURES_OBSERVED_SELECTION]


# In[20]:


Xval_tenfeat = Xval[TEN_BEST_FEATURES_OBSERVED_SELECTION]


# In[21]:


statsmodel_logit(Xtrain_tenfeat, Ytrain, Xval_tenfeat, Yval)


# In[85]:


model = model_logit_sklearn(Xtrain_tenfeat, Ytrain, Xval_tenfeat, Yval)


# In[88]:


model_file_path = os.path.join(data_dir, "step5\\model.joblib")
dump(model, model_file_path)


# In[24]:


xtrain_balanced, ytrain_balanced = balancing_data_with_SMOTE(Xtrain_tenfeat,
                                                             Ytrain)


# In[29]:


GridSearchCV_forBestParams(xtrain_balanced, ytrain_balanced)


# In[91]:


lgbm_model(xtrain_balanced, ytrain_balanced, Xval_tenfeat, Yval)


# In[ ]:
