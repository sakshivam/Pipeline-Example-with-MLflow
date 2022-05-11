#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import os
import pickle
from joblib import dump, load
from sklearn.metrics import accuracy_score


# In[18]:


MISSING_VAL_COLUMNS = ['car']
MISSING_VAL_COLUMNS_TO_FILL = ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']

DROP_COL_LIST_WITH_ONE_CLASS = ['toCoupon_GEQ5min']
TEN_BEST_FEATURES_OBSERVED_SELECTION = ['passanger', 'coupon','CoffeeHouse', 'destination', 'expiration', 'toCoupon_GEQ25min',
                                        'Bar', 'gender', 'Restaurant20To50','temperature' ]


# In[12]:


dict_for_clubbing = {"RestaurantLessThan20": {'1~3':'1~8' , '4~8': '1~8' },
                     "CarryAway": {'1~3':'1~8ngt8','4~8': '1~8ngt8', 'gt8':'1~8ngt8', 
                                   'less1': 'less1_never','never': 'less1_never'},
                     "CoffeeHouse": {'1~3': '1~8ngt8','4~8': '1~8ngt8', 'gt8': '1~8ngt8'},
                     "Bar": {'1~3':'1~8','4~8':'1~8'},
                     "income": {'$12500 - $24999': '$12500-$62499','$25000 - $37499': '$12500-$62499',
                                '$37500 - $49999': '$12500-$62499','$50000 - $62499': '$12500-$62499',
                                '$75000 - $87499':'$62499-$99999','$87500 - $99999':'$62499-$99999',
                                '$62500 - $74999':'$62499-$99999'},
                     "occupation": {'Architecture & Engineering': 'Arch_cons_Health_Food_Farm', 
                                   'Construction & Extraction': 'Arch_cons_Health_Food_Farm',
                                   'Healthcare Support': 'Arch_cons_Health_Food_Farm',
                                   'Food Preparation & Serving Related': 'Arch_cons_Health_Food_Farm',
                                   'Healthcare Practitioners & Technical':'Arch_cons_Health_Food_Farm', 
                                   'Farming Fishing & Forestry': 'Arch_cons_Health_Food_Farm', 
                                    'Business & Financial':'Arts_Comm_Building_Buss_Edu_Retired', 
                                    'Education&Training&Library':'Arts_Comm_Building_Buss_Edu_Retired', 
                                    'Retired':'Arts_Comm_Building_Buss_Edu_Retired', 
                                    'Arts Design Entertainment Sports & Media':'Arts_Comm_Building_Buss_Edu_Retired',
                                    'Community & Social Services':'Arts_Comm_Building_Buss_Edu_Retired', 
                                    'Building & Grounds Cleaning & Maintenance':'Arts_Comm_Building_Buss_Edu_Retired',
                                   'Computer & Mathematical':'comp_Inst_Pers_Sales_Unemployed', 
                                    'Installation Maintenance & Repair':'comp_Inst_Pers_Sales_Unemployed', 
                                    'Personal Care & Service':'comp_Inst_Pers_Sales_Unemployed', 
                                    'Sales & Related':'comp_Inst_Pers_Sales_Unemployed', 
                                    'Unemployed':'comp_Inst_Pers_Sales_Unemployed',
                                   'Student':'Stud_manage_off_Trans_Protect_Life_Prod',
                                    'Management':'Stud_manage_off_Trans_Protect_Life_Prod',
                                    'Office & Administrative Support':'Stud_manage_off_Trans_Protect_Life_Prod',
                                    'Transportation & Material Moving':'Stud_manage_off_Trans_Protect_Life_Prod',
                                    'Protective Service':'Stud_manage_off_Trans_Protect_Life_Prod',
                                    'Life Physical Social Science':'Stud_manage_off_Trans_Protect_Life_Prod',
                                    'Production Occupations':'Stud_manage_off_Trans_Protect_Life_Prod'},
                     "education": {'Bachelors degree':'Bach_Mast_Asso',
                                   'Associates degree':'Bach_Mast_Asso', 
                                   'Graduate degree (Masters or Doctorate)':'Bach_Mast_Asso', 
                                   'Some college - no degree':'Nodeg_HSGrad',
                                   'High School Graduate':'Nodeg_HSGrad'},
                     "maritalStatus": {'Married partner':'Married_Divorced_Widowed',
                                       'Divorced':'Married_Divorced_Widowed',
                                       'Widowed':'Married_Divorced_Widowed', 
                                       'Single':'Single_Unmarriedpartner',
                                       'Unmarried partner':'Single_Unmarriedpartner'},
                     "age": {'26':'below26','21':'below26', 'below21':'below26',
                             '31':'above26','50plus':'above26', '36':'above26','41':'above26', '46':'above26'},
                     "coupon": {'Restaurant(20-50)':'Rest(20-50)_n_Bar', 'Bar':'Rest(20-50)_n_Bar',
                                'Restaurant(<20)':'Rest(<20)_CH_CT',
                                'Coffee House':'Rest(<20)_CH_CT',
                                'Carry out & Take away':'Rest(<20)_CH_CT'},
                     "destination": {'Home':'Home_n_Work','Work':'Home_n_Work'},
                     "passanger": {'Alone':'Alone_n_Kids','Kid(s)':'Alone_n_Kids'},
                    "weather": {'Snowy':'Snowy_n_Rainy','Rainy':'Snowy_n_Rainy'},
                     "time": {'7AM':'7AM_n_10PM','10PM':'7AM_n_10PM', 
                              '6PM':'6PM_10AM_2PM','10AM':'6PM_10AM_2PM', '2PM':'6PM_10AM_2PM'},  
                }


# In[13]:


dict_to_get_ordinal_features = {"destination": {"No Urgent Place": 2,"Home_n_Work": 1 },
                "passanger":   {"Alone_n_Kids": 3, "Partner": 2, "Friend(s)": 1},
                "weather": {"Sunny": 2,"Snowy_n_Rainy": 1},
                "time": {"6PM_10AM_2PM": 2,"7AM_n_10PM": 1},
                "coupon": {"Rest(<20)_CH_CT": 2,"Rest(20-50)_n_Bar": 1},
                "expiration": {"1d": 2,"2h": 1},
                "gender": {"Male": 2,"Female": 1},
                "age": {"below26": 2,"above26": 1},
                "maritalStatus": {"Single_Unmarriedpartner": 2,"Married_Divorced_Widowed": 1},
                "education": {"Some High School": 3,"Nodeg_HSGrad": 2, "Bach_Mast_Asso": 1 },
                "occupation": {"Arch_cons_Health_Food_Farm": 5,"Stud_manage_off_Trans_Protect_Life_Prod": 4,"comp_Inst_Pers_Sales_Unemployed": 3, "Arts_Comm_Building_Buss_Edu_Retired": 2, "Legal": 1 },
                "income": {"Less than $12500": 4,"$12500-$62499": 3,"$100000 or More": 2, "$62499-$99999": 1},
                "car": {"Car that is too old _n_Mazda5": 4,"Scooter and motorcycle": 3, "do not drive": 2, "crossover": 1 },
                "Bar": {"1~8": 4,"less1": 3, "gt8": 2, "never": 1 },
                "CoffeeHouse": {"1~8ngt8": 3, "less1": 2, "never": 1 },
                "CarryAway": {"1~8ngt8": 2, "less1_never": 1 },
                "RestaurantLessThan20": {"gt8": 4,"1~8": 3, "never": 2, "less1": 1 },
                "Restaurant20To50": {"gt8": 5, "4~8": 4,"1~3": 3, "less1": 2, "never": 1 }
                }


# In[2]:


current_path = os.getcwd()
dirname, filename = os.path.split(current_path)
print(dirname, filename)


# In[3]:


data_dir= os.path.join(dirname, 'files')
print(data_dir)


# In[4]:


test_file_path = os.path.join(data_dir, "step1\\test.parquet")


# In[5]:


testdf= pd.read_parquet(test_file_path)


# In[7]:


testdf.head()


# ## Analysis Functions

# In[ ]:





# ## Action Functions

# In[9]:


def Drop_Missing_Val_Columns(df, col_list_to_drop):
    df.drop(columns=col_list_to_drop, inplace=True)
    return df


# In[10]:


def Fill_missing_Val_Columns(df, col_list_to_fill):
    for i in col_list_to_fill:
        df[i] = df[i].fillna(df[i].value_counts().index[0])
            #print(df[i].isna().sum())
    return df


# In[15]:


def replace_values_using_dict(df, dict_to_replace_values):
    df = df.replace(dict_to_replace_values)
    return df


# In[21]:


def split_into_XnY(df, dep_col):
    Ytrain = df[dep_col]
    Xtrain = df.drop([dep_col], axis=1)
    return Xtrain, Ytrain


# In[38]:


def predictions(testdf, dep_col, features, model_file_path):
    Xtest, Ytest = split_into_XnY(testdf, dep_col)
    Xtest_tenfeat = Xtest[features]
    mdl = load(model_file_path)
    y_pred = mdl.predict(Xtest_tenfeat)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = ['Y_Predicted']
    merged_df = pd.concat([Xtest,y_pred_df], axis=1)
    return merged_df


# ## Flow

# In[11]:


testdf = Fill_missing_Val_Columns(testdf, MISSING_VAL_COLUMNS_TO_FILL)
testdf = Drop_Missing_Val_Columns(testdf, MISSING_VAL_COLUMNS)


# In[16]:


testdf=replace_values_using_dict(testdf,dict_for_clubbing )


# In[17]:


testdf=replace_values_using_dict(testdf,dict_to_get_ordinal_features )


# In[20]:


model_file_path = os.path.join(data_dir, "step5\\model.joblib")


# In[39]:


result_df = predictions(testdf, 'Y', TEN_BEST_FEATURES_OBSERVED_SELECTION)
result_df


# In[40]:


predicted_df_file_path = os.path.join(data_dir, "step6\\predicted_data.parquet")
result_df.to_parquet(predicted_df_file_path, index=False)

