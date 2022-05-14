#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import os


# In[35]:


current_path = os.getcwd()
dirname, filename = os.path.split(current_path)
print(dirname, filename)


# In[36]:


data_dir= os.path.join(dirname, 'files')
print(data_dir)


# In[37]:


train_file_path = os.path.join(data_dir, "step1\\train.parquet")


# In[38]:


#traindf= pd.read_csv(train_file_path)
traindf= pd.read_parquet(train_file_path)


# In[39]:


traindf.head()


# ## Analysis Functions

# In[40]:


def Columns_to_drop(df):
    #Find out which columns have null values and total number of Nan values in that column
    # Determine percentage of missing data in a particular column for cleaning of the Data. 
    # Columns in data with more than 95% null should be dropped.
    null_count = df.isna().sum()
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_val_columns =list(percent_missing[percent_missing > 95].index)
    return missing_val_columns


# In[41]:


def Missing_Val_Columns_to_fill(df):
    #Finding the value counts for columns with percent_missing between 0 and 95%
    #Replacing Nan values in each of these columns with most frequent value in that column.
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_val_col_list_to_fill = list(percent_missing[(percent_missing > 0) & (percent_missing < 95)].index)
    
    return missing_val_col_list_to_fill


# ## Action Functions

# In[42]:


def Drop_Missing_Val_Columns(df, col_list_to_drop):
    df.drop(columns=col_list_to_drop, inplace=True)
    return df


# In[43]:


def Fill_missing_Val_Columns(df, col_list_to_fill):
    for i in col_list_to_fill:
        df[i] = df[i].fillna(df[i].value_counts().index[0])
            #print(df[i].isna().sum())
    return df


# ## Flow

# In[44]:


MISSING_VAL_COLUMNS = Columns_to_drop(traindf)
print(MISSING_VAL_COLUMNS)


# In[45]:


MISSING_VAL_COLUMNS_TO_FILL = Missing_Val_Columns_to_fill(traindf)
print(MISSING_VAL_COLUMNS_TO_FILL)


# In[46]:


traindf = Fill_missing_Val_Columns(traindf, MISSING_VAL_COLUMNS_TO_FILL)


# In[47]:


traindf = Drop_Missing_Val_Columns(traindf, MISSING_VAL_COLUMNS)


# In[48]:


traindf_cleaned= traindf


# In[49]:


train_cleaned_file_path = os.path.join(data_dir, "step2\\train_cleaned.parquet")
traindf_cleaned.to_parquet(train_cleaned_file_path, index=False)


# ## Remarks 

# ### CONFIG

# In[50]:


#MISSING_VAL_COLUMNS = ['car']
#MISSING_VAL_COLUMNS_TO_FILL = ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']


# In[ ]:




