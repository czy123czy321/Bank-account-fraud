#!/usr/bin/env python
# coding: utf-8

# feature engineering to prepare training data for XGBoost and logistic regression.
# 1. Build a switcher where I can turn on and off each feature engineering step. In the notebook. should I still build the feature engineering steps on top of each other? i.e. ohe on imputed df.
# 2. Outliers: We don't want to remove it, leaving them untouched for XGBoost. However, logistic regression is more sensitive to outliers, so we need to handle it.
# 3. Missing values: XGBoost is good at dealing missing values. However, missing values are still imputed to prepare for other algorithm and compare model performance. 
# 4. Scaling: scale the data with robust scaler because there are significant outliers, still scaling is not very helpful for xgboost but we add it case we are trying other algorithms.
# 5. Mutual information evaluation after all the preprocessing to find significant input features. 
# 6. Build a base model evalution for the dataset, which is evaluated by recall@5% because it's highly imbalanced. 

# # 1. Setup

# In[1]:


print("Starting setups")
import sys
import os

sys.path.append(os.path.abspath("feature_engineering"))


# In[2]:


import yaml
with open('../params.yaml', 'r') as file:
    params = yaml.safe_load(file)
params


# In[3]:


dir_path = os.getcwd()
parent_dir = os.path.dirname(dir_path)
data_folder = parent_dir+params['data_location']

print('Data is stored at', data_folder)


# In[4]:


with open(dir_path+"/feature_engineering/feature_flag.yaml", "r") as file:
    config = yaml.safe_load(file)["feature_engineering"]

print(f"Current configuration for feature engineering is: {config}")


# ## 1.1 Import libraries and reading data

# In[5]:


print("Importing packages and reading data...")
import pandas as pd
from  preprocessing import *

pd.set_option('display.max_columns', 500)

import warnings as wr
wr.filterwarnings('ignore')


# In[6]:


df_base = pd.read_csv(f"{data_folder}/Base_backup.csv", header=0)
df = df_base.copy()


# ## 1.2 Drop features with no variance

# In[7]:


constant_feature =[]
for x in df.columns:
    if df[x].nunique() == 1:
        constant_feature.append(x)
print("Dropping the constant features:", constant_feature)       
df = drop_columns(df, df[constant_feature])


# ### 1.3 Change the dataype of binary features into type boolean

# In[8]:


print("Changing the dataype of binary features into type boolean")
binary_features = df.columns[df.nunique() == 2].tolist()

binary_features.remove('source')
# Convert these features"to boolean
df[binary_features] = df[binary_features].astype(bool)

# Verify changes
print(df[binary_features].dtypes)


# ## 1.3 Train test split

# In[10]:


print("splitting train/test sets")
y = df['fraud_bool']
X = df.drop(columns=['fraud_bool'], axis = 1)

from sklearn.model_selection import train_test_split

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

print(f"Training set (X_train) before feature engineering: {X_train.shape}")
print(f"Test set (X_test) before feature engineering: {X_test.shape}")
print(f"Training set (y_train) before feature engineering: {y_train.shape}")
print(f"Test set (y_test) before feature engineering: {y_test.shape}")

categorical_features, numerical_features = split_num_cat(df)
print('Categorical features before feature engineering:', categorical_features)
print('Numerical features before feature engineering:', numerical_features)


# ## 2. Switcher

# In[ ]:


print("Start feature engineering steps...")


# In[ ]:


if config["impute"]:
    print("Applying imputation for missing values...")
    X_train, X_test = impute_missing_values(X_train, X_test)
else:
    print("Not applying imputation.")


# In[ ]:


if config["one_hot_encoding"]:
    print("Applying one-hot encoding...")
    X_train, X_test = one_hot_encode(X_train, X_test)
else:
    print("Not applying one hot encoder.")


# In[ ]:


if config["binning"]:
    print("Applying binning...")
    X_train, X_test = bin_bank_months_count(X_train, X_test, y_train)
else:
    print("Not applying binning.")


# In[ ]:


if config["robust_scaler"]:
    print("Applying robust scaling...")
    X_train, X_test = robust_scaler(X_train, X_test)
else:
    print("Not applying scaler.")


# In[ ]:


if config["outlier_handling"]:
    print("Removing outliers...")
    X_train, y_train = handle_outliers(X_train, y_train)
else:
    print("Not handling outliers")


# In[ ]:


if config["smote"]:
    print("Perfoming SMOTE to handle class imbalance")
    X_train, y_train = smote(X_train, y_train, over_ratio=0.7, under_ratio=0.9)
else:
    print("Not applying SMOTE")


# In[ ]:


if config["mutual_information"]:
    print("Calculating mutual information scores based on the final training set")
    mi_scores = mutual_information(X_train, y_train)
else:
    print("Not calculating mutua information")
    
if config["chi2_test"]:
    print("Calculating chi2 based on the final training set")
    chi2_results = chi2_test(X_train, y_train)
else:
    print("Not calculating chi2")


# # Save the final output 

# In[ ]:


print("Exporting final dataset...")
export_final_df(X_train, y_train, X_test, y_test)


# In[ ]:





# In[ ]:




