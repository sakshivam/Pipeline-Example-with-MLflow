#!/usr/bin/env python
# coding: utf-8

# ## Splitting dataset into train and test files

# In[16]:


pip install fastparquet


# In[1]:


import pandas as pd
import os


# In[2]:


current_path = os.getcwd()
print(current_path)


# In[4]:


dirname, filename = os.path.split(current_path)
print(dirname, filename)


# In[7]:


data_dir= os.path.join(dirname, 'files')
print(data_dir)


# In[8]:


data_file_path = os.path.join(data_dir, 'in-vehicle-coupon-recommendation.csv')


# In[9]:


coupondf=pd.read_csv(data_file_path)
fraction_train = 0.7


# In[10]:


coupondf.head()


# In[11]:


def test_train_split(df, frac_train):
    frac= frac_train
    train_df = df.sample(frac = 0.7)
    test_df = df.drop(train_df.index)
    return train_df, test_df 


# In[12]:


traindf, testdf = test_train_split(coupondf, fraction_train)


# In[26]:


train_file_path = os.path.join(data_dir, "step1\\train.parquet")


# In[27]:


traindf.to_parquet(train_file_path, index=False)


# In[ ]:


#train_file_path = os.path.join(dirname, "train.csv")


# In[38]:


#traindf.to_csv(train_file_path, index=False)


# In[28]:


test_file_path = os.path.join(data_dir, "step1\\test.parquet")


# In[29]:


testdf.to_parquet(test_file_path, index=False)


# In[ ]:




