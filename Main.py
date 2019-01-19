import DataLoader
import Doao
from EvaluationClassifiers import get_decision_tree_model, compute_accuracy


def main():
    root_path = r'C:\Users\micha\Downloads\Nt1'
    all_data_file_path = 'train_Loan'
    data_file_path = r'{}\{}.csv'.format(root_path, all_data_file_path)
    columns_types_dict = {'Loan_ID': 'Categorical',
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

    data_set = DataLoader.load_data(data_file_path, columns_types)
    train_set, test_set = DataLoader.split_train_test(data_set, 0.2)
    pair_classifiers = Doao.build_pair_classifiers(train_set)

    dt_score = get_decision_tree_model(train_set, test_set)

    print_df = data_set.copy()
    # print_df['DOAO_Predicted_class'] = -1
    predicted_classification = []
    for index, instance in test_set.iterrows():
        classification = Doao.classify_new_instance(instance, pair_classifiers)
        predicted_classification.append(classification)
        print_df.loc[[index], 'DOAO_Predicted_class'] = classification
        # print('Instance.num[{}]: classifies as: [{}]'.format(list(instance), classification[0]))
    acc = compute_accuracy(test_set.iloc[:, -1], predicted_classification)
    print("Accuracy of DOAO model: {}".format(acc))
    print("Accuracy of DT model: {}".format(dt_score))
    print_df.to_csv(r"Data\DOAO_predictions.csv", index=False)


if __name__ == '__main__':
    main()
