import DOAO
import DataLoader
from EvaluationClassifier import get_decision_tree_model, compute_accuracy


def main():
    root_path = r'.\ExampleDatasets'
    data_name1 = 'train_Loan'
    data_file_path1 = r'{}\{}.csv'.format(root_path, data_name1)
    columns_types_dict1 = {'Loan_ID': 'Categorical',
                           'Gender': 'Categorical',
                           'Married': 'Categorical',
                           'Dependents': 'Categorical',
                           'Education': 'Categorical',
                           'Self_Employed': 'Categorical',
                           'ApplicantIncome': 'Numeric',
                           'CoapplicantIncome': 'Numeric',
                           'LoanAmount': 'Numeric',
                           'Loan_Amount_Term': 'Numeric',
                           'Credit_History': 'Categorical',
                           'Property_Area': 'Categorical',
                           'Loan_Status': 'Categorical'}
    data_name2 = 'iris'
    data_file_path2 = r'{}\{}.csv'.format(root_path, data_name2)
    columns_types_dict2 = {'A1': 'Numeric',
                           'A2': 'Numeric',
                           'A3': 'Numeric',
                           'A4': 'Numeric',
                           'Class': 'Object'}
    run(data_name1, data_file_path1, columns_types_dict1)
    run(data_name2, data_file_path2, columns_types_dict2)


def run(data_name, data_file_path, columns_types_dict):
    print("===========================")
    print("Running DOAO classifier on: {}".format(data_name))
    predicted_classification = []
    data_set = DataLoader.load_data(data_file_path, columns_types_dict)
    print_df = data_set.copy()
    train_set, test_set = DataLoader.split_train_test(data_set, 0.2)
    pair_classifiers = DOAO.build_pair_classifiers(train_set)

    for index, instance in test_set.iterrows():
        classification = DOAO.classify_new_instance(instance, pair_classifiers)[0]
        predicted_classification.append(classification)
        print_df.loc[[index], 'DOAO_Predicted_class'] = classification
        print('Instance.num[{}]: classifies as: [{}]'.format(index, classification))
    acc = compute_accuracy(test_set.iloc[:, -1], predicted_classification)
    dt_score = get_decision_tree_model(train_set, test_set)
    print("Accuracy of DOAO model: {}".format(acc))
    print("Accuracy of DT model: {}".format(dt_score))
    print_df.to_csv(r"Outputs/{}_DOAO_predictions.csv".format(data_name), index=False)


if __name__ == '__main__':
    main()
