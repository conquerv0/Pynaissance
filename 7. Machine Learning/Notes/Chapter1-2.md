### Chapter 1, A Primer

__1. General Model Categories__: 

- Classification: Predict discrete values
- Regression: Predict continuous values. 
- Multi-classification: Sample can have multiple labels. 
- Clustering: Unsupervised learning, training samples does not have labels

__2. Training, Validation, and Testing__

- Training set is used to train the model, i.e. calculate variables from parameters. 
- Validation set is used to fine-tune the model's hyperparameter
- Testing is used to validate a model's generalization characteristics.

### Chapter 2, Model Evaluation and Selection

#### I. Evaluation Metrics and General Concept

__1. Error__

- Training error, empirical error: error in predication made by the model in training set
- Generalization error: error in predication made by the model in out-of-sample new data set.

__2. Overfitting, Underfitting__

- Overfitting: 
- Underfitting:

#### II. Model Evaluation Case

__1. Hold-out__

~~~python
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

x, y = datasets.load_iris(return_X_Y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print(pd.value_count(y_train))

print(pd.value_counts(y_test))
~~~


__2. Cross Validation__

~~~python
from sklearn.model_selection import KFold
import numpy as np

X = np.random.randint(1, 6, 66)
kf = KFold(n_splits = 6)

for train, test in kf.split(X):
  print(train, '\n', test)
~~~

__3. Leave-one-out LOO__

Leave-ont-out method is a special case in cross validation. We assume Dataset D contain some m samples, let k=m, and each subset contain one sample. This means LOO is not subject to random splitting since there only exists one way to split m subset. This means, the model performance evaluated under LOO is vastly similar to the expected training result of the model on dataset D. Such, this evaluation method is considered accurate, despite its apparent flaw: for large dataset, the computational cost of training m model is huge. 

~~~python
from sklearn.model_selection import LeaveOneOut
import numpy as np

X = np.random.randint(1, 10, 5)
kf = LeaveOneOut()
for train, test in kf.split(X):
  print(train, '\n', test)

