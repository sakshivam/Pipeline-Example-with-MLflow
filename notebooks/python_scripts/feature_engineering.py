#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import os
import numpy as np


# In[22]:


current_path = os.getcwd()
dirname, filename = os.path.split(current_path)
print(dirname, filename)


# In[23]:


data_dir= os.path.join(dirname, 'files')
print(data_dir)


# In[24]:


train_cleaned_file_path = os.path.join(data_dir, "step2\\train_cleaned.parquet")
traindf_cleaned= pd.read_parquet(train_cleaned_file_path)


# In[25]:


traindf_cleaned.head()


# ## Analysis Functions

# In[26]:


def Categorical_Numerical_Features_split(df):
    categorical_data = df.select_dtypes(exclude=[np.number])
    cat_list = list(categorical_data.columns)
    numeric_data = df.select_dtypes(include=[np.number])
    num_list = list(numeric_data.columns)
    return cat_list, num_list


# In[27]:


def plot_stackedgraph_categorical(df, cat_list):
    for i in cat_list:
        df.groupby([i])['Y'].apply(lambda x: x.value_counts() / len(x)).transpose().unstack().plot(kind='bar',stacked = True)


# In[28]:


def scatterplot_for_numerical_features(df, num_lst):
    num_list_new = num_lst[0:7]
    for i in num_list_new:
        ax1 =df.plot.scatter(x=i, y='Y',c='DarkBlue')


# ## Action Functions

# In[29]:


def replace_values_using_dict(df, dict_to_replace_values):
    df = df.replace(dict_to_replace_values)
    return df


# In[ ]:





# ## Flow

# In[30]:


categorical_list, numeric_list = Categorical_Numerical_Features_split(traindf_cleaned)
print("List of categorical Features:\n", categorical_list, "\n")
print("List of Numerical Features:\n", numeric_list)


# In[31]:


#plot_stackedgraph_categorical(traindf_cleaned, categorical_list)


# ## Remarks

# ### CONFIG

# In[32]:


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


# In[33]:


traindf_cleaned=replace_values_using_dict(traindf_cleaned,dict_for_clubbing )


# In[34]:


#traindf_cleaned['time'].value_counts()


# In[35]:


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


# In[36]:


traindf_cleaned=replace_values_using_dict(traindf_cleaned,dict_to_get_ordinal_features )


# In[37]:


#scatterplot_for_numerical_features(traindf_cleaned, numeric_list)


# In[38]:


traindf_with_feature_engg= traindf_cleaned


# In[39]:


train_fengg_file_path = os.path.join(data_dir, "step3\\traindf_with_feature_engg.parquet")
traindf_with_feature_engg.to_parquet(train_fengg_file_path, index=False)


# ## Remarks

# ### CONFIG

# In[40]:


# Categorical_Features = ['destination', 'passanger', 'weather', 'time', 'coupon', 'expiration', 'gender', 'age', 'maritalStatus',
#                         'education', 'occupation', 'income', 'Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20',
#                         'Restaurant20To50']
# Numerical Features = ['temperature', 'has_children', 'toCoupon_GEQ5min', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min',
#                       'direction_same', 'direction_opp', 'Y']

# dict_for_clubbing = {"RestaurantLessThan20": {'1~3':'1~8' , '4~8': '1~8' },
#                      "CarryAway": {'1~3':'1~8ngt8','4~8': '1~8ngt8', 'gt8':'1~8ngt8', 
#                                    'less1': 'less1_never','never': 'less1_never'},
#                      "CoffeeHouse": {'1~3': '1~8ngt8','4~8': '1~8ngt8', 'gt8': '1~8ngt8'},
#                      "Bar": {'1~3':'1~8','4~8':'1~8'},
#                      "income": {'$12500 - $24999': '$12500-$62499','$25000 - $37499': '$12500-$62499',
#                                 '$37500 - $49999': '$12500-$62499','$50000 - $62499': '$12500-$62499',
#                                 '$75000 - $87499':'$62499-$99999','$87500 - $99999':'$62499-$99999',
#                                 '$62500 - $74999':'$62499-$99999'},
#                      "occupation": {'Architecture & Engineering': 'Arch_cons_Health_Food_Farm', 
#                                    'Construction & Extraction': 'Arch_cons_Health_Food_Farm',
#                                    'Healthcare Support': 'Arch_cons_Health_Food_Farm',
#                                    'Food Preparation & Serving Related': 'Arch_cons_Health_Food_Farm',
#                                    'Healthcare Practitioners & Technical':'Arch_cons_Health_Food_Farm', 
#                                    'Farming Fishing & Forestry': 'Arch_cons_Health_Food_Farm', 
#                                     'Business & Financial':'Arts_Comm_Building_Buss_Edu_Retired', 
#                                     'Education&Training&Library':'Arts_Comm_Building_Buss_Edu_Retired', 
#                                     'Retired':'Arts_Comm_Building_Buss_Edu_Retired', 
#                                     'Arts Design Entertainment Sports & Media':'Arts_Comm_Building_Buss_Edu_Retired',
#                                     'Community & Social Services':'Arts_Comm_Building_Buss_Edu_Retired', 
#                                     'Building & Grounds Cleaning & Maintenance':'Arts_Comm_Building_Buss_Edu_Retired',
#                                    'Computer & Mathematical':'comp_Inst_Pers_Sales_Unemployed', 
#                                     'Installation Maintenance & Repair':'comp_Inst_Pers_Sales_Unemployed', 
#                                     'Personal Care & Service':'comp_Inst_Pers_Sales_Unemployed', 
#                                     'Sales & Related':'comp_Inst_Pers_Sales_Unemployed', 
#                                     'Unemployed':'comp_Inst_Pers_Sales_Unemployed',
#                                    'Student':'Stud_manage_off_Trans_Protect_Life_Prod',
#                                     'Management':'Stud_manage_off_Trans_Protect_Life_Prod',
#                                     'Office & Administrative Support':'Stud_manage_off_Trans_Protect_Life_Prod',
#                                     'Transportation & Material Moving':'Stud_manage_off_Trans_Protect_Life_Prod',
#                                     'Protective Service':'Stud_manage_off_Trans_Protect_Life_Prod',
#                                     'Life Physical Social Science':'Stud_manage_off_Trans_Protect_Life_Prod',
#                                     'Production Occupations':'Stud_manage_off_Trans_Protect_Life_Prod'},
#                      "education": {'Bachelors degree':'Bach_Mast_Asso',
#                                    'Associates degree':'Bach_Mast_Asso', 
#                                    'Graduate degree (Masters or Doctorate)':'Bach_Mast_Asso', 
#                                    'Some college - no degree':'Nodeg_HSGrad',
#                                    'High School Graduate':'Nodeg_HSGrad'},
#                      "maritalStatus": {'Married partner':'Married_Divorced_Widowed',
#                                        'Divorced':'Married_Divorced_Widowed',
#                                        'Widowed':'Married_Divorced_Widowed', 
#                                        'Single':'Single_Unmarriedpartner',
#                                        'Unmarried partner':'Single_Unmarriedpartner'},
#                      "age": {'26':'below26','21':'below26', 'below21':'below26',
#                              '31':'above26','50plus':'above26', '36':'above26','41':'above26', '46':'above26'},
#                      "coupon": {'Restaurant(20-50)':'Rest(20-50)_n_Bar', 'Bar':'Rest(20-50)_n_Bar',
#                                 'Restaurant(<20)':'Rest(<20)_CH_CT',
#                                 'Coffee House':'Rest(<20)_CH_CT',
#                                 'Carry out & Take away':'Rest(<20)_CH_CT'},
#                      "destination": {'Home':'Home_n_Work','Work':'Home_n_Work'},
#                      "passanger": {'Alone':'Alone_n_Kids','Kid(s)':'Alone_n_Kids'},
#                     "weather": {'Snowy':'Snowy_n_Rainy','Rainy':'Snowy_n_Rainy'},
#                      "time": {'7AM':'7AM_n_10PM','10PM':'7AM_n_10PM', 
#                               '6PM':'6PM_10AM_2PM','10AM':'6PM_10AM_2PM', '2PM':'6PM_10AM_2PM'},  
#                 }

# dict_to_get_ordinal_features = {"destination": {"No Urgent Place": 2,"Home_n_Work": 1 },
#                 "passanger":   {"Alone_n_Kids": 3, "Partner": 2, "Friend(s)": 1},
#                 "weather": {"Sunny": 2,"Snowy_n_Rainy": 1},
#                 "time": {"6PM_10AM_2PM": 2,"7AM_n_10PM": 1},
#                 "coupon": {"Rest(<20)_CH_CT": 2,"Rest(20-50)_n_Bar": 1},
#                 "expiration": {"1d": 2,"2h": 1},
#                 "gender": {"Male": 2,"Female": 1},
#                 "age": {"below26": 2,"above26": 1},
#                 "maritalStatus": {"Single_Unmarriedpartner": 2,"Married_Divorced_Widowed": 1},
#                 "education": {"Some High School": 3,"Nodeg_HSGrad": 2, "Bach_Mast_Asso": 1 },
#                 "occupation": {"Arch_cons_Health_Food_Farm": 5,"Stud_manage_off_Trans_Protect_Life_Prod": 4,"comp_Inst_Pers_Sales_Unemployed": 3, "Arts_Comm_Building_Buss_Edu_Retired": 2, "Legal": 1 },
#                 "income": {"Less than $12500": 4,"$12500-$62499": 3,"$100000 or More": 2, "$62499-$99999": 1},
#                 "car": {"Car that is too old _n_Mazda5": 4,"Scooter and motorcycle": 3, "do not drive": 2, "crossover": 1 },
#                 "Bar": {"1~8": 4,"less1": 3, "gt8": 2, "never": 1 },
#                 "CoffeeHouse": {"1~8ngt8": 3, "less1": 2, "never": 1 },
#                 "CarryAway": {"1~8ngt8": 2, "less1_never": 1 },
#                 "RestaurantLessThan20": {"gt8": 4,"1~8": 3, "never": 2, "less1": 1 },
#                 "Restaurant20To50": {"gt8": 5, "4~8": 4,"1~3": 3, "less1": 2, "never": 1 }
#                 }


# In[ ]:




