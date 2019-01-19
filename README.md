# Multi_Class_Binary_Classification_Using_DOAO_Classifier
Implementation of the Diversified One-Against-One (DOAO) classifier, proposed by Kang et al.,2015.
The implementaion was tested on two multiclass datasets:
* Iris: Class{Iris-setosa, Iris-versicolor, Iris-virginica}
* BankLoan: Property_Area{Urban, Semiurban, Rural}

**Paper reference**: 
Kang, Seokho, Sungzoon Cho, and Pilsung Kang. "Constructing a multi-class classifier using one-against-one approach with different binary classifiers." Neurocomputing 149 (2015): 677-682.
 
The Diversified One-Against-One (DOAO) classifier finds the best classification algorithm for each class pair when
applying the one-against-one approach to multi-class classification problems.

Procedure DOAO:
1. C <- Init classifier set
2. For each class pair (i, j) do:
  - 2.1: Init the set of datapoints whose class labels are i or j
  - 2.2: Train cnadidate classifiers, each of which is trained using a different algorithm.
  - 2.3: Obtain validation error for each candidate classifier
  - 2.4: cl <- Find the calassifier that corresponds to the minimum valdation error.
  - 2.5  Add cl to the classifiers set C
3.	end procedure
