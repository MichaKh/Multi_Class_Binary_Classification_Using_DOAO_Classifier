import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def check_for_missing_data(df):
    cleaned_df = df  # check if its by ref or it copy
    if cleaned_df.empty:
        return None
    none_column_data = cleaned_df.columns[cleaned_df.isnull().any()]
    for col in none_column_data:
        col_type = cleaned_df[col].dtype
        if col_type in [np.float64, np.int64]:  # numeric
            cleaned_df.loc[:, col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        else:  # categorical value
            cleaned_df.loc[:, col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0])
    return cleaned_df


def load_data(file_path, types_dict):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, header=0)
    df = change_types(df, types_dict)
    cleaned_data = check_for_missing_data(df)
    return cleaned_data


def split_train_test(df, frac):
    train_set, test_set = train_test_split(df, test_size=frac, random_state=42)
    encoded_test = encode_categorical_features(test_set)
    encoded_train = encode_categorical_features(train_set)
    return encoded_train , encoded_test


def change_types(data_frame, types):
    for col_name, t in types.items():
        if t == 'Categorical':
            data_frame[col_name] = data_frame[col_name].astype('category')
        if t == 'Class':
            data_frame[col_name] = data_frame[col_name].astype('object')
    return data_frame


def encode_categorical_features(X):
    encoded_X = X.copy()
    le = LabelEncoder()
    for col in encoded_X.columns:
        if encoded_X[col].dtype not in [np.float64, np.int64]:  # numeric
            encoded_X[col] = le.fit_transform(X[col])
    return encoded_X
