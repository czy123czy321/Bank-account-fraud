#!/usr/bin/env python
# coding: utf-8

# # Helper functions

# In[1]:


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

def drop_columns(df, column_name):
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


# # 2. Handle missing values¶

# XGBoost supports missing values by default. In tree algorithms, branch directions for missing values are learned during training. And the gblinear booster treats missing values as zeros.
# 
# We still choose to handle missing values in case of we are using any other types of models. 
# Also for features with high percentage of missing values, we choose to add an additional column to indicate where the value is missing, so the model can still learn from missing values. 

# In[2]:


def impute_missing_values(X_train, X_test):
    """
    Imputes missing values in X_train and X_test based on predefined rules.

    - Replaces `-1` values with `NaN`.
    - Median imputation for specified columns.
    - Bank months count missing values replaced with `0` and tracked in a new indicator column.
    - Previous address months count imputed based on address bucket medians.

    Parameters:
    X_train (pd.DataFrame): Training dataset
    X_test (pd.DataFrame): Test dataset

    Returns:
    pd.DataFrame, pd.DataFrame: Imputed training and test datasets
    """

    # Identify columns with missing values represented by -1
    missing_value_cols = [col for col in X_train.columns if (X_train[col].min() == -1)]
    print("Features with missing values represented by -1:")
    print(missing_value_cols)

    # Replace -1 values with NaN
    X_train[missing_value_cols] = X_train[missing_value_cols].replace(-1, np.nan)
    X_test[missing_value_cols] = X_test[missing_value_cols].replace(-1, np.nan)

    # Create copies to avoid modifying the original data
    df_train_imputed = X_train.copy()
    df_test_imputed = X_test.copy()

    # Step 1: Median Imputation for selected columns
    columns_to_impute = ['current_address_months_count', 'session_length_in_minutes', 'device_distinct_emails_8w']

    medians = df_train_imputed[columns_to_impute].median()

    for column in columns_to_impute:
        df_train_imputed[column] = df_train_imputed[column].fillna(medians[column])
        df_test_imputed[column] = df_test_imputed[column].fillna(medians[column])

    # Step 2: Bank months count - fill missing with 0 and track missing values
    df_train_imputed['bank_months_count_was_missing'] = df_train_imputed['bank_months_count'].isna().astype(int)
    df_test_imputed['bank_months_count_was_missing'] = df_test_imputed['bank_months_count'].isna().astype(int)

    df_train_imputed['bank_months_count'] = df_train_imputed['bank_months_count'].fillna(0)
    df_test_imputed['bank_months_count'] = df_test_imputed['bank_months_count'].fillna(0)

    # Step 3: Bucket-Based Imputation for prev_address_months_count
    bin_edges = pd.cut(df_train_imputed['current_address_months_count'], 12, retbins=True)[1]

    df_train_imputed['current_address_bucket'] = pd.cut(df_train_imputed['current_address_months_count'], bins=bin_edges)
    df_test_imputed['current_address_bucket'] = pd.cut(df_test_imputed['current_address_months_count'], bins=bin_edges)

    # Compute medians for each bucket
    bucket_medians = df_train_imputed.groupby('current_address_bucket')['prev_address_months_count'].median()

    # Indicator column for missing values
    df_train_imputed['prev_address_months_count_was_missing'] = df_train_imputed['prev_address_months_count'].isna().astype(int)
    df_test_imputed['prev_address_months_count_was_missing'] = df_test_imputed['prev_address_months_count'].isna().astype(int)

    # Train set: fill missing using bucket median
    df_train_imputed['prev_address_months_count'] = df_train_imputed.groupby('current_address_bucket')['prev_address_months_count'].transform(
        lambda x: x.fillna(x.median())
    )

    # Test set: apply training bucket medians
    df_test_imputed['prev_address_months_count'] = df_test_imputed.apply(
        lambda row: bucket_medians[row['current_address_bucket']]
        if pd.isna(row['prev_address_months_count']) and row['current_address_bucket'] in bucket_medians
        else row['prev_address_months_count'],
        axis=1
    )

    # Drop temporary bucket column
    df_train_imputed.drop(columns=['current_address_bucket'], inplace=True)
    df_test_imputed.drop(columns=['current_address_bucket'], inplace=True)

    # Validate that all missing values are handled
    print("Train set null values after imputation:")
    print(df_train_imputed[missing_value_cols].isna().sum())

    print("\nTest set null values after imputation:")
    print(df_test_imputed[missing_value_cols].isna().sum())
    print(f"Training set shape after imputation: {df_train_imputed.shape}")
    print(f"Test set shape after imputation: {df_test_imputed.shape}")

    return df_train_imputed, df_test_imputed


# ## 3. One-hot encode for categorical features

# In[3]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(X_train, X_test):
    """
    Applies one-hot encoding to categorical features in the dataset.

    - Identifies categorical features (dtype = 'object')
    - Uses OneHotEncoder to transform categorical features
    - Merges encoded columns with numerical features
    - Ensures consistent encoding across train and test sets

    Parameters:
    X_train (pd.DataFrame): Training dataset
    X_test (pd.DataFrame): Test dataset

    Returns:
    pd.DataFrame, pd.DataFrame: One-hot encoded training and test datasets
    """

    # Identify categorical features
    categorical_features = [col for col in X_train.columns if X_train[col].dtypes == 'object']
    print("Categorical features to encode:", categorical_features)

    if not categorical_features:
        print("No categorical features found. Skipping one-hot encoding.")
        return X_train, X_test

    # Initialize one-hot encoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Encode training and test sets
    ohe_X_train = pd.DataFrame(ohe.fit_transform(X_train[categorical_features]))
    ohe_X_test = pd.DataFrame(ohe.transform(X_test[categorical_features]))

    # Set index to match original data
    ohe_X_train.index = X_train.index
    ohe_X_test.index = X_test.index

    # Set column names to match encoded feature names
    ohe_feature_names = ohe.get_feature_names_out(input_features=categorical_features)
    ohe_X_train.columns = ohe_feature_names
    ohe_X_test.columns = ohe_feature_names

    # Drop categorical features and concatenate with encoded data
    num_X_train = X_train.drop(categorical_features, axis=1)
    num_X_test = X_test.drop(categorical_features, axis=1)

    X_train_encoded = pd.concat([num_X_train, ohe_X_train], axis=1)
    X_test_encoded = pd.concat([num_X_test, ohe_X_test], axis=1)

    print(f"One-hot encoded training shape: {X_train_encoded.shape}")
    print(f"One-hot encoded test shape: {X_test_encoded.shape}")

    return X_train_encoded, X_test_encoded


# # 4. Binning

# Binning the bank_months_count, to turn it into a categorical variable for lower cardinality. 
# 
# Don't want to bin any other features as we don't want to lose details of the data 

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

def bin_bank_months_count(X_train, X_test, y_train):
    """
    Bins the 'bank_months_count' feature into custom intervals and replaces them with pre-defined medians.

    - First bin includes only 0, the rest are grouped in intervals of 4
    - Applies binning to both training and test datasets
    - Plots fraud proportion against binned values
    - Drops the original 'bank_months_count' column after transformation

    Parameters:
    X_train (pd.DataFrame): Training dataset
    X_test (pd.DataFrame): Test dataset
    y_train (pd.Series): Target variable for training set (needed for visualization)

    Returns:
    pd.DataFrame, pd.DataFrame: Training and test datasets with binned 'bank_months_count'
    """

    # Make copies to avoid modifying original datasets
    X_train_binned = X_train.copy()
    X_test_binned = X_test.copy()

    min_val = X_train_binned["bank_months_count"].min()
    print(f"the minimum value of bank_months_count is {min_val}")
    # Define bins: First bin starts from min_val, and remaining are in intervals of 4
    bins = [min_val, min_val + 1] + [i for i in range(int(min_val) + 5, 37, 4)]
    
    median_labels = [0, 2.5, 6.5, 10.5, 14.5, 18.5, 22.5, 26.5, 30.5]  # Median of each bin

    print("\n Bin Ranges:")
    for i in range(len(bins) - 1):
        print(f"Bin {i+1}: [{bins[i]}, {bins[i+1]}) -> Median: {median_labels[i]}")

    # Apply binning transformation
    X_train_binned["bank_months_count_binned"] = pd.cut(
        X_train_binned["bank_months_count"], bins=bins, labels=median_labels, include_lowest=True, right=False
    ).astype(float)

    X_test_binned["bank_months_count_binned"] = pd.cut(
        X_test_binned["bank_months_count"], bins=bins, labels=median_labels, include_lowest=True, right=False
    ).astype(float)

    # Display bin distribution
    bin_counts = X_train_binned["bank_months_count_binned"].value_counts().sort_index()
    print("\n Bin medians and counts in training set:\n")
    print(bin_counts)

    # Visualization: Fraud proportion per bin
    X_train_binned["fraud_bool"] = y_train  # Temporarily add fraud labels for visualization
    fraud_proportion = X_train_binned.groupby("bank_months_count_binned")["fraud_bool"].mean()

    plot_data = (
        X_train_binned[X_train_binned["fraud_bool"] == 1]["bank_months_count_binned"]
        .value_counts(normalize=True)
        .sort_index()
    )

    # Plot fraud proportion per bin
    plt.figure(figsize=(8, 4))
    plot_data.plot(kind="bar", color="#fc8d62", edgecolor="black")
    plt.xlabel("Bank Months Count (Binned)")
    plt.ylabel("Proportion of Fraud (fraud_bool = 1)")
    plt.title("Proportion of Fraud by Bank Months Count (Binned)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Remove temporary fraud column
    X_train_binned.drop(columns=["fraud_bool"], inplace=True)

    # Drop original 'bank_months_count' after binning
    X_train_binned.drop(columns="bank_months_count", inplace=True)
    X_test_binned.drop(columns="bank_months_count", inplace=True)

    print(f"Final shape after binning - Train: {X_train_binned.shape}, Test: {X_test_binned.shape}")

    return X_train_binned, X_test_binned


# # 5. Scaling

# In[5]:


from sklearn.preprocessing import RobustScaler

def robust_scaler(X_train, X_test):
    """
    Apply robust scaler on numerical features. Robust scaler scale the data based on IQR and median, 
    which maeks the dataset more robust against outliers. https://proclusacademy.com/blog/robust-scaler-outliers/ 

    - First split the dataset into numerical and categorical features. 
    - Then import RobustScaler from sklearn
    - Fit robust scaler on the training set, then transform numerical_features in both training and test set. 
    - Add a 'scaled_' prefix to the columns that are scaled, then drop the original column.

    Parameters:
    X_train (pd.DataFrame): Training dataset
    X_test (pd.DataFrame): Test dataset

    Returns:
    pd.DataFrame, pd.DataFrame: Training and test datasets with scaled numerical features
    """
    categorical_features, numerical_features = split_num_cat(X_train)
    print('Categorical features:', categorical_features)
    print('Numerical features:', numerical_features)
    
    X_train[numerical_features].describe()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
     
    robust_scaler = RobustScaler()
    
    scaled_train = robust_scaler.fit_transform(X_train_scaled[numerical_features])
    scaled_test = robust_scaler.transform(X_test_scaled[numerical_features])
    
    # add new columns scaled features while keeping the original feature with the unscaled values. 
    for i, feature in enumerate(numerical_features):
        X_train_scaled['scaled_' + feature] = scaled_train[:, i]
        X_test_scaled['scaled_' + feature] = scaled_test[:, i]
    
    print(X_train_scaled.describe())
    
    # drop the original columns before scaling:
    X_train_scaled.drop(columns=numerical_features, inplace=True)
    X_test_scaled.drop(columns=numerical_features, inplace=True)

    print(f"Final shape after binning - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled


# help with unsupervised learning, minimise bias against one variabel 

# # 6. Handle outliers - not used

# In[6]:


def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

def handle_outliers(X_train, y_train):
    categorical_features, numerical_features = split_num_cat(X_train)
    print('Categorical features:', categorical_features)
    print('Numerical features:', numerical_features)

    # Detect outliers in numerical features
    total_outliers = X_train[numerical_features].apply(detect_outliers).sum()
    outlier_percentage = (total_outliers / len(X_train)) * 100
    print("Percentage of outliers per numerical variable\n", outlier_percentage)

    # For all fraud cases, calculate the percentage of outliers
    X_train_fraud_only = X_train[y_train == 1].copy()
    total_outlier_fraud_only = detect_outliers(X_train_fraud_only[numerical_features]).sum()
    outlier_percentages_fraud_only = (total_outlier_fraud_only / len(X_train_fraud_only)) * 100
    print("Percentage of outliers among the fraud cases (fraud_bool = 1):")
    print(outlier_percentages_fraud_only)

    # Calculate fraud outlier percentages per feature
    outlier_fraud_percentages = {}
    for feature in numerical_features:
        outliers = detect_outliers(X_train[feature])
        fraud_outliers = X_train.loc[outliers & (y_train == 1), feature]
        percentage_fraud_outliers = (len(fraud_outliers) / len(X_train[X_train[feature].notna()])) * 100
        outlier_fraud_percentages[feature] = percentage_fraud_outliers
    print("Percentage of outliers that are fraud:")
    print(outlier_fraud_percentages)

    # Remove outliers using the 1st and 99th percentile
    q1 = X_train[numerical_features].quantile(0.01)
    q99 = X_train[numerical_features].quantile(0.99)
    X_train_cleaned = X_train[(X_train[numerical_features] >= q1).all(axis=1) &
                              (X_train[numerical_features] <= q99).all(axis=1)]
    y_train_cleaned = y_train.loc[X_train_cleaned.index].copy()

    # Print percentage of retained data
    percentage_retained = (len(X_train_cleaned) / len(X_train)) * 100
    print(f"Percentage of data retained after dropping outliers: {percentage_retained:.2f}%")

    # Calculate percentage of fraud cases retained
    fraud_indices = y_train[y_train == 1].index
    fraud_retained = len(fraud_indices.intersection(X_train_cleaned.index))
    percentage_fraud_retained = (fraud_retained / len(fraud_indices)) * 100
    print(f"Percentage of fraud_bool = 1 retained in cleaned data: {percentage_fraud_retained:.2f}%")
    
    print(f"Final X_train shape: {X_train_cleaned.shape}")
    print(f"Final y_train shape: {y_train_cleaned.shape}")

    return X_train_cleaned, y_train_cleaned


# # Handle imbalance with SMOTE, only on the training set

# In[7]:


from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def smote(X_train, y_train, over_ratio=0.8, under_ratio=1.0):
    """
    Applies SMOTE for oversampling and RandomUnderSampler for undersampling.
    
    :param X_train: Feature matrix.
    :param y_train: Target vector.
    :param over_ratio: Oversampling ratio for the minority class (default: 0.8).
    :param under_ratio: Undersampling ratio for the majority class after oversampling (default: 1.0).
    :return: Resampled X_train and y_train.
    """
    print(f"Before SMOTE: {Counter(y_train)}")
    print(f"Before SMOTE, shape of the training set: {X_train.shape}")

    # Oversample the minority class
    over = SMOTE(sampling_strategy=over_ratio, random_state=2)
    X_train_res, y_train_res = over.fit_resample(X_train, y_train)

    print(f"After oversampling: {Counter(y_train_res)}")

    # Undersample the majority class
    under = RandomUnderSampler(sampling_strategy=under_ratio, random_state=2)
    X_train_resampled, y_train_resampled = under.fit_resample(X_train_res, y_train_res)

    print(f"After undersampling: {Counter(y_train_resampled)}")
    print(f"After undersampling, shape of the training set: {X_train_resampled.shape}")

    return X_train_resampled, y_train_resampled


# ## Mutual Information and chi2

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, chi2, SelectKBest

def mutual_information(X_train, y_train):
    """
    Compute Mutual Information scores for all features.
    """
    X = X_train.copy()
    y = y_train.copy()

    # Label encode categorical features
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()

    # Determine discrete features
    discrete_features = X.dtypes == int

    # Compute Mutual Information scores
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns).sort_values(ascending=False)

    print("Mutual Information Scores:")
    print(mi_scores)

    def plot_mi_scores(scores):
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")
        plt.show()

    plt.figure(dpi=100, figsize=(8, 16))
    plot_mi_scores(mi_scores)

    return mi_scores


def chi2_test(X_train, y_train):
    """
    Perform Chi-Square test for categorical features.
    """
    categorical_features, numerical_features = split_num_cat(X_train)
    X_cat = X_train[categorical_features].copy()
    chi2_test = SelectKBest(score_func=chi2).fit(X_cat, y_train)

    chi2_output = pd.DataFrame({
        'feature': X_cat.columns,
        'chi2_score': chi2_test.scores_,
        'p_value': chi2_test.pvalues_
    }).sort_values(by=['p_value'])

    print("Chi-Square Test Results:")
    print(chi2_output)

    # Filter significant features
    chi2_output_significant = chi2_output[chi2_output['p_value'] <= 0.05]

    # Plot significant chi2 scores
    plt.figure(figsize=(15, 10))
    sns.barplot(data=chi2_output_significant, x='chi2_score', y='feature')
    plt.title("Significant Chi-Square Scores")
    plt.show()

    return chi2_output


# MI can help you to understand the relative potential of a feature as a predictor of the target, considered by itself.
# It's possible for a feature to be very informative when interacting with other features, but not so informative all alone. MI can't detect interactions between features. It is a univariate metric.
# The actual usefulness of a feature depends on the model you use it with. A feature is only useful to the extent that its relationship with the target is one your model can learn. Just because a feature has a high MI score doesn't mean your model will be able to do anything with that information. You may need to transform the feature first to expose the association.

# # Finalize data for training

# In[9]:


def export_final_df(X_train, y_train, X_test, y_test, data_folder):
    bool_features = [col for col in X_train.columns if X_train[col].dtypes == 'bool']
    X_train[bool_features] = X_train[bool_features].astype("int")
    y_train = y_train.astype("int")
    X_test[bool_features] = X_test[bool_features].astype("int")
    y_test = y_test.astype("int")

    print(f'Final X_train shape {X_train.shape}')
    print(f'Final y_train shape {y_train.shape}')
    print(f'Final X_test shape {X_test.shape}')
    print(f'Final y_test shape {y_test.shape}')

    print("Feature engineering is done. Exporting the final training data and test data to location:", data_folder)
    
    X_train.to_csv(data_folder + "/x_train_data.csv", index=True)
    X_test.to_csv(data_folder + "/x_test_data.csv")
    
    y_train.to_csv(data_folder + "/y_train_data.csv")
    y_test.to_csv(data_folder + "/y_test_data.csv")
    print("Data successfully exported into csv!")

