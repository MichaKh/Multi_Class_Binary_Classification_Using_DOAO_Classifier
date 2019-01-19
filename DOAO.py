import os
from collections import OrderedDict
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def load_data(data_file_path, columns_types_dict):
    if not os.path.exists(data_file_path):
        return None  # Please check your provided file path
    data_df = pd.read_csv(data_file_path, header=0)
    data_df = change_columns_types(data_df, columns_types_dict)
    clean_data_df = check_for_missing_values(data_df)
    return clean_data_df


def change_columns_types(data_df, columns_types_dict):
    for col, t in columns_types_dict.items():
        if t == 'Categorical':
            data_df[col] = data_df[col].astype('category')
    return data_df


def check_for_missing_values(data_df):
    clean_data_df = data_df
    if not data_df.empty:
        nan_columns_data = list(data_df.columns[data_df.isnull().any()])
        for nan_column in nan_columns_data:
            col_dtype = data_df[nan_column].dtype
            if col_dtype in [np.float64, np.int64]:  # column is numeric
                clean_data_df.loc[:, nan_column] = data_df[nan_column].fillna(data_df[nan_column].mean())
            else:  # column is categorical
                clean_data_df.loc[:, nan_column] = data_df[nan_column].fillna(data_df[nan_column].mode().iloc[0])
        # null_data = data_df.loc[data_df.isnull().any(axis=1)]
        # if not null_data.empty():
        #     data_df.fillna(data_df.mean())
    return clean_data_df


# public
def classify_new_instance(instance, pair_classifiers):
    arr_votes = []
    for classifier in pair_classifiers:
        class_value = classifier.predict(instance)
        arr_votes.append(class_value)
    return max(arr_votes, key=arr_votes.count)


# public
def build_pair_classifiers(data_set):
    classifiers_set = get_classifiers()
    class_labels = list(data_set.iloc[:, -1].unique())
    domains_num = len(class_labels)
    pair_classifiers_set = []
    for i in range(domains_num):
        for j in range(i + 1, domains_num):
            curr_pair_data_points = instances_selection(class_labels[i], class_labels[j], data_set)
            curr_pair_classifiers = build_subset_pair_classifiers(curr_pair_data_points, classifiers_set)
            curr_best_classifier = choose_best_classifier(curr_pair_classifiers)
            pair_classifiers_set.append(curr_best_classifier)
    return pair_classifiers_set


def get_classifiers():
    """
    Chooses the models that will be tested with the implemented DOAO classifier
    """
    classifiers = {'Decision Tree': DecisionTreeClassifier(max_depth=5),
                   'Naive Bayes': MultinomialNB(),
                   'Logistic Regression': LogisticRegression()}
    return classifiers


'''
choose only the instances which classification is c1 or c2
'''


def instances_selection(c1, c2, data_set):
    return data_set.loc[data_set[data_set.columns[-1]].isin([c1, c2])]


'''
train m models form classifiers_set of size m on dataset data_set
'''


def build_subset_pair_classifiers(data_set, classifiers_dict):
    le = preprocessing.LabelEncoder()
    models_scores = {}
    kf = KFold(n_splits=10)
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]
    for column_name in X.columns:
        if X[column_name].dtype not in [np.float64, np.int64, np.int, np.float]:
            X[column_name] = le.fit_transform(X[column_name])
    for name, classifier in classifiers_dict.items():
        model = classifier.fit(X, y)
        scores = cross_val_score(model, X, y, cv=kf)
        avg_score = np.mean(scores)
        models_scores[name] = avg_score
    return models_scores


'''
choose the best classifer based on accuracy from pair_classifiers which contain m trained models
'''


def choose_best_classifier(pair_classifiers_scores):
    max_classifier, max_score = OrderedDict(sorted(pair_classifiers_scores.items(), key=itemgetter(1)))
    return max_classifier
