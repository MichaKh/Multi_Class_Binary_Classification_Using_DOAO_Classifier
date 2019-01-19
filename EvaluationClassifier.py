from sklearn import tree
from sklearn.metrics import accuracy_score

from DOAO import encode_categorical_features


def get_decision_tree_model(train_df, test_df):
    classifier = tree.DecisionTreeClassifier()
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, 0:-1]
    y_test = test_df.iloc[:, -1]
    X_train = encode_categorical_features(X_train)
    X_test = encode_categorical_features(X_test)
    trained_classifier = classifier.fit(X_train, y_train)
    predicted_labels = trained_classifier.predict(X_test.values)
    return compute_accuracy(y_test, predicted_labels)


def compute_accuracy(actual_set, actual_predicted):
    return accuracy_score(actual_set, actual_predicted)
