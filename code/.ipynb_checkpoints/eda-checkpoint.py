#!/usr/bin/env python
# coding: utf-8

# ### Summary of Interesting Finding from Exploratory data analysis
# *  The target variable fraud_bool is highly imbalanced (1% fraud, 99% non-fraud). This imbalance suggst that normalization, resampling techniques, or model adjustments (e.g., using class weights) may be necessary in subsequent steps.
# *  Missing values are represented as -1, which could skew statistical results. It could be worth to impute missing values using appropriate methods (e.g., mean or mode) and create new columns to suggest where the value is missing.
# *  Missing values in prev_address_months_count and bank_months_count are higher in fraud cases. Investigate the relationship between fraud type and missing values; explore whether fraudsters are less likely to provide certain information.
# *  Violin plot suggests outliers in some variables, which may need investigation or removal. Analyze whether outliers represent rare fraud cases; decide whether to keep or remove them based on fraud association.
# *  The violin plot shows significant differences in the distribution of features like credit_risk_score, date_of_birth_distinct_emails_4w, current_address_months_count, name_email_similarity, and prev_address_months_countdays between fraud and non-fraud cases.
# *  Some variables like source, foreign_request, phone_mobile_valid, device_distinct_emails_8w have highly imbalanced distributions (e.g., mostly "INTERNET" for source). However, when class imbalance is adjusted, these variables show some associations with fraud cases, therefore we should keep them for modelling instead of removal.
# *  Some categorical variables like customer age, payment_type, employment_status, housing_status, keep_session_alive, phone_mobile_valid, proposed_credit_limit, foreign_request, source, device_os, and device_distinct_emails_8w show a disproportionately high fraud rate in a certain class, which could signal a stronger association with fraud after accounting for imbalance in class representation.
# *  No significant correlations were found between numerical variables in multivariate analysis. 

# # 1. Import libraries and reading data

# In[1]:


import yaml
with open ("/Users/zoe/Documents/Bank-account-fraud/params.yaml") as p:
    params = yaml.safe_load(p)


# In[2]:


params


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.style.use('ggplot')

from pylab import rcParams

import seaborn as sns
sns.set_style('whitegrid')
my_palette = sns.color_palette("Paired")

from scipy.stats import chi2_contingency, pearsonr

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import chi2

pd.set_option('display.max_columns', 500)

import warnings as wr
wr.filterwarnings('ignore')


# In[4]:


df_base = pd.read_csv(params['data_location'])
df = df_base.copy()


# In[5]:


df.shape


# In[6]:


from IPython.display import display
pd.options.display.max_columns = None
display(df.head(10))


# In[7]:


df.describe().transpose()


# In[8]:


df.info()


# In[9]:


df.nunique()


# In[10]:


print(f"Number of duplicate records in the data: {df.duplicated().sum()}")


# # 2. Helper functions

# In[11]:


# split categorical and numerical features
def split_num_cat(df):
    """
    Function to split columns into two, one having categorical features and another having numerical feautures
    Parameters
    ----------
    df : dataframe
            pass in full dataframe
    ----------
    Returns: 
        a list of categorical features
        a list of numerical features
    """
    categorical_features = []
    numerical_features = []

    for x in df.columns: 
        if df[x].nunique() > 12:
            numerical_features.append(x)
        elif df[x].nunique() >=2:
            categorical_features.append(x)

    if 'fraud_bool' in categorical_features:
        categorical_features.remove('fraud_bool')
    
    return categorical_features, numerical_features

def drop_unnecessary_columns(df, column_name):
    """
    Function to delete the list of columns 
    Parameters
    ----------
    df : dataframe
            pass in full dataframe
    column_name : list
            pass in list of full column
    ----------
    Returns: Dataframe
    """
    df = df.drop(column_name, axis=1)
    return df


# ## 3. Preliminary data analysis

# #### 3.1 Drop features with no variance

# In[12]:


constant_feature =[]
for x in df.columns:
    if df[x].nunique() == 1:
        constant_feature.append(x)

print("Constant features:", constant_feature)


# In[13]:


df = drop_unnecessary_columns(df, df[constant_feature])


# #### 3.2 Split categorical and numerical features

# In[14]:


categorical_features, numerical_features = split_num_cat(df)
print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)


# #### 3.3 Distribution of the target feature

# In[15]:


df_fraud_only = df[df['fraud_bool'] == 1]
df_non_fraud_only = df[df['fraud_bool'] == 0]
print("Proportion of 'fraud' vs 'not fraud' {:>3.2f}%".format(len(df_fraud_only)/len(df_non_fraud_only)*100))


# In[16]:


fraud_vals = pd.DataFrame(df['fraud_bool'].value_counts())
print(fraud_vals)


# #### 3.4 Proportion of missing values

# In[17]:


missing_value_val = [x for x in df_base.columns if (df_base[x].min() == -1)]

print("Features with missing values represented by -1:")
print(missing_value_val)


# In[18]:


missing_vals = pd.DataFrame()

missing_features = ['prev_address_months_count', 'current_address_months_count', 'bank_months_count', 'session_length_in_minutes', 'device_distinct_emails_8w']

for feature in missing_features:
    df.loc[df[feature] < 0, feature] = np.nan # df[feature] = df[feature].replace(-1, np.nan)
    missing_vals_col = df.groupby('fraud_bool')[feature].apply(lambda x: round(x.isna().sum()/len(x) * 100, 2))
    missing_vals[feature] = missing_vals_col
    
missing_vals = pd.DataFrame(missing_vals.T.stack())

missing_vals.reset_index(inplace=True)
missing_vals.rename(columns={'level_0': 'feature', 0: 'missing_vals'}, inplace=True)

print(missing_vals)


# ## 4. Exploratory analysis

# ### 4.1 Target variable - fraud_bool

# In[19]:


sns.countplot(x='fraud_bool', data=df)


# ### 4.2 Numerical Variables

# In[20]:


print("Numerical features:", numerical_features)


# #### 4.2.1 Distribution of each numerical feature in terms of the target value in KDE plot 

# In[21]:


# Create a grid of subplots
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))

# Add a title to the figure
fig.suptitle('Distribution of Numeric Features by Fraud Status')

# Loop through the numeric features and plot a kernel density plot for each feature
for i, feature in enumerate(numerical_features):
    ax = axes[i // 3][i % 3]
    sns.kdeplot(data=df[df['fraud_bool'] == 0][feature], fill=True, ax=ax, label='Not Fraud')
    sns.kdeplot(data=df[df['fraud_bool'] == 1][feature], fill=True, ax=ax, label='Fraud')
    ax.set_xlabel(feature)
    ax.legend()

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()


# #### 4.2.2 Violin Plots - the median, quartiles, and outliers, split by target value.

# In[22]:


fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
fig.suptitle('Violin Plot of Numeric Features by Fraud Status')

for i, feature in enumerate(numerical_features):
    ax = axes[i // 3][i % 3]
    sns.violinplot(data=df, x='fraud_bool', y=feature, ax=ax, palette='Paired')
    ax.set_xlabel('')
    ax.set_ylabel(feature)
    ax.set_xticklabels(['Not Fraud', 'Fraud'])

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()


# ####  4.2.3 Scatterplot of numerical features against fraud type

# In[ ]:





# ####  4.2.4 Multivariate Analysis - correlation matrix plot

# In[23]:


df_corr = df_base[numerical_features].corr()
df_corr


# In[24]:


plt.figure(figsize=(10, 10))
# sns.heatmap(df_corr, annot=True, cmap='Pastel2')
sns.heatmap(df_corr, xticklabels=df_corr.columns.values,  annot=True, yticklabels = df_corr.columns.values)


# ### 4.3 Categorical Variables

# In[25]:


print("Categorical features:", categorical_features)


# #### 4.3.1 Countplots of categorical features

# In[26]:


cols = 3
rows = 6

fig, axes = plt.subplots(rows, cols, figsize=(16, 24))
fig.suptitle("Value Counts of Categorical Features", fontsize=16)
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
    df[feature].value_counts().plot(kind='bar', ax=axes[idx], title=f"{feature}", color='#8da0cb' )
    axes[idx].set_xlabel("Categories")
    axes[idx].set_ylabel("Count")

for idx in range(len(categorical_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# #### 4.3.2 Proportion of Categorical Features by fraud type

# In[27]:


fig, axes = plt.subplots(6, 3, figsize=(16, 24))
axes = axes.flatten()
fig.suptitle('Distribution of Categorical Features by fraud type (Proportion)', fontsize=16)

for idx, feature in enumerate(categorical_features):
    # Calculate the proportion of fraud_bool within each category
    feature_counts = df[feature].value_counts(normalize=True).sort_index()
    plot_data = df.groupby(feature)['fraud_bool'].value_counts(normalize=True).unstack(fill_value=0)
    
    plot_data.plot(kind='bar', stacked=True, color=['#93cb8d', '#fc8d62'], ax=axes[idx])
    axes[idx].set_title(f'{feature}')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('Proportion')
    axes[idx].tick_params(axis='x')
    axes[idx].legend(title='Fraud type', loc='lower left', bbox_to_anchor=(0, 0))

for idx in range(len(categorical_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()


# #### 4.3.3 Proportion of fraud cases (i.e.fraud_bool = 1) by class in each categorical feature

# In[28]:


fig, axes = plt.subplots(6, 3, figsize=(16, 24))
axes = axes.flatten()
fig.suptitle('Proportion of only fraud (i.e.fraud_bool = 1) by class (relative to total data)', fontsize=16)

for idx, feature in enumerate(categorical_features):
    # Calculate the proportion of fraud_bool = 1 for each category relative to the total data
    plot_data = (
        df[df['fraud_bool'] == 1][feature].value_counts() / df[feature].value_counts()
    ).sort_index()
    
    # Plot the proportion for fraud_bool = 1
    plot_data.plot(kind='bar', color='#fc8d62', ax=axes[idx])
    axes[idx].set_title(f'{feature}')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('Proportion (Fraud Only)')
    axes[idx].tick_params(axis='x')

# Remove extra subplots
for idx in range(len(categorical_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()


# #### 4.3.4 Adjusted proportion of fraud cases (scaled by the overall class frequency) for each category within the feature

# In[29]:


target = 'fraud_bool'

# Set up a grid of subplots
rows, cols = 6, 3
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 30))
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
    # Create the fraud-by-bucket distribution
    fraud_by_bucket_distribution = pd.crosstab(df[feature], df[target], normalize=False)
    fraud_by_bucket_proportions = fraud_by_bucket_distribution.div(fraud_by_bucket_distribution.sum(axis=1), axis=0)

    # Calculate feature class proportions
    class_distribution = df[feature].value_counts(normalize=False).sort_index()
    class_proportions = class_distribution / class_distribution.sum()

    # Calculate target class proportions
    fraud_distribution = df[target].value_counts(normalize=True) 

    # Scale the fraud proportions by class distribution
    scaled_fraud_proportions = fraud_by_bucket_proportions.div(fraud_distribution, axis=1).mul(class_proportions, axis=0)
    scaled_fraud_proportions

    # Extract fraud_bool = 1 proportions
    scaled_fraud_by_class = scaled_fraud_proportions[1]

    # Plot
    ax = axes[idx]
    scaled_fraud_by_class.plot(kind='bar', ax=ax, color='#fce562', alpha=0.8)
    ax.set_title(f'{feature}')
    ax.set_xlabel('')
    ax.set_ylabel('Scaled Proportion')
    ax.tick_params(axis='x', rotation=45)

# Remove empty subplots if the number of features is less than 6x3
for idx in range(len(categorical_features), len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.suptitle('Scaled Fraud Proportions by Categorical Features', fontsize=20)
plt.show()


# ### 4.4 Ordinal variables

# In[30]:


# Select categorical features with more than 2 unique values and float or integer data type
ordinal_categorical_features = [
    x for x in categorical_features
    if df_base[x].dtype in ['float64', 'int64'] and df_base[x].nunique() >2
]

print("Ordinal Categorical Features are:")
print(ordinal_categorical_features)


# ## 5. Data preprocessing

# ### 5.1 Imputation for missing values¶

# In[31]:


# Defining pipelines 
from sklearn.base import BaseEstimator, TransformerMixin

df_imputed = df.replace(-1, np.nan).copy()

categorical_features, numerical_features = split_num_cat(df)

class GroupImputerWithMissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, mode_columns, mean_columns, grouping_column):
        self.mode_columns = mode_columns
        self.mean_columns = mean_columns
        self.grouping_column = grouping_column

    def fit(self, X, y=None):
        # Compute mode for mode_columns
        self.group_modes = {}
        for column in self.mode_columns:
            self.group_modes[column] = (
                X.loc[X[column] != -1].groupby(self.grouping_column)[column].apply(lambda x: x.mode()[0] if not x.mode().empty else x.median())
            )
        # Compute mean for mean_columns
        self.group_means = {}
        for column in self.mean_columns:
            self.group_means[column] = (
                X.loc[X[column] != -1].groupby(self.grouping_column)[column].mean()
            )
        return self

    def transform(self, X):
        X = X.copy()
        
        # Handle mode_columns
        X[self.mode_columns] = X[self.mode_columns].replace(-1, np.nan)
        for column in self.mode_columns:
            X[column] = X.groupby(self.grouping_column)[column].transform(
                lambda x: x.fillna(self.group_modes[column].get(x.name, np.nan))
            )

        # Handle mean_columns and add missing indicators
        X[self.mean_columns] = X[self.mean_columns].replace(-1, np.nan)
        for column in self.mean_columns:
            missing_indicator_col = column + '_was_missing'
            X[missing_indicator_col] = (X[column].isna()).astype(int)
            X[column] = X.groupby(self.grouping_column)[column].transform(
                lambda x: x.fillna(self.group_means[column].get(x.name, np.nan))
            )
        
        return X


# In[32]:


missing_value_val = [x for x in df_base.columns if (df_base[x].min() == -1)]

print("Features with missing values represented by -1:")
print(missing_value_val)


# In[33]:


from sklearn.pipeline import Pipeline

# Define the columns and pipeline

# For features with missing_vals_percentage < 1, impute missing values with mode of the group 
mode_columns = ['current_address_months_count', 'session_length_in_minutes', 'device_distinct_emails_8w']

# For features with higher missing_vals_percentages, impute missing values with mean of the group, 
# then add a column that indicate where the value was missing
mean_columns = ['bank_months_count', 'prev_address_months_count']

grouping_column = 'fraud_bool'

pipeline = Pipeline(steps=[
    ('group_imputer_with_indicator', GroupImputerWithMissingIndicator(
        mode_columns=mode_columns,
        mean_columns=mean_columns,
        grouping_column=grouping_column
    ))
])

# Apply the pipeline
df_imputed = pipeline.fit_transform(df_imputed)

# Verify the transformation
print("Count of -1 values after imputation:")
print((df_imputed[mode_columns + mean_columns] == -1).sum())

print("Counts in '_was_missing' columns:")
print(df_imputed[['bank_months_count_was_missing', 'prev_address_months_count_was_missing']].value_counts())


# In[34]:


df_imputed.describe().transpose()


# ### 5.2 One-hot encode for categorical features

# In[35]:


df_new = df_imputed.copy()

# One-hot encoding for categorical featuers with dtype as 'object'
object_features = [col for col in df_new.columns if df_new[col].dtypes == 'object']
print(object_features)


# In[36]:


df_new = pd.DataFrame(pd.get_dummies(df_new, prefix=object_features, dtype=np.int64))


# In[37]:


df_new.describe().transpose()


# ## 6. Chi-squared test for processed categorical features 

# In[38]:


categorical_features = [c for c in df_new.columns if df_new[c].nunique() >= 2 and df_new[c].nunique() <= 12]
categorical_features.remove('fraud_bool')
print(categorical_features)

df_cat = df_new[categorical_features].copy()

# from sklearn.feature_selection import chi2
import scipy.stats as stats
chi_squared_values = {}
for feature in categorical_features:
    contingency_table = pd.crosstab(df_cat[feature], df_new['fraud_bool'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Store the Chi-Squared value
    chi_squared_values[feature] = chi2

# Convert the dictionary to a DataFrame for easy plotting
chi_squared_df = pd.DataFrame(list(chi_squared_values.items()), columns=['Feature', 'Chi-Squared Value'])

# Plot the Chi-Squared values
plt.figure(figsize=(10, 6))
sns.barplot(x='Chi-Squared Value', y='Feature', data=chi_squared_df, color='#93cb8d')
plt.title('Chi-Squared Value by Categorical Feature')
plt.xlabel('Chi-Squared Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

