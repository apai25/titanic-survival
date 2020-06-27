import numpy as np 
import pandas as pd 

# =========================================TRAINING======================================
# Importing the training dataset
train = pd.read_csv('train.csv')

# Creating numpy arrays for independent variables x_train and dependent variable y_train
x_train = train.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y_train = train.iloc[:, 1].values

# Cleaning x_train by imputing missing values with either mean(for floats) or fill_value(for strings)
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train[:, 2:3] = mean_imputer.fit_transform(x_train[:, 2:3])
x_train[:, 5:6] = mean_imputer.fit_transform(x_train[:, 5:6])

word_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x_train[:, 6:7] = word_imputer.fit_transform(x_train[:, 6:7])

# OneHotEncoding categorical variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencoder = OneHotEncoder(drop='first', dtype=int)
ct = ColumnTransformer([('encoding', onehotencoder, [0, 1, 6])], remainder='passthrough')
x_train = ct.fit_transform(x_train)

# Scaling data (using .toarray() because x_train is a sparse matrix)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)

# Creating and training the classifier
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(x_train, y_train)

# =========================================TESTING======================================
# Importing the testing dataset
test = pd.read_csv('test.csv')

# Creating numpy array for the necessary testing columns
x_test = test.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values
ids = test.iloc[:, 0].values

# Cleaning x_test by imputing missing values with mean(for floats)
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_test[:, 2:3] = mean_imputer.fit_transform(x_test[:, 2:3])
x_test[:, 5:6] = mean_imputer.fit_transform(x_test[:, 5:6])

# OneHotEncoding categorical variables
x_test = ct.transform(x_test)

# Scaling data
x_test = sc_x.transform(x_test)

# Predicting with the classifier
predictions = classifier.predict(x_test)

# Reshaping ids for concatenation
ids = ids.reshape(len(ids), 1)

# Reshaping predictions for concatenation
predictions = predictions.reshape(len(predictions), 1)

# Cocatenating passenger ids and predictions
submission = np.concatenate((ids, predictions), axis=1)

# Saving submission to submission.csv
np.savetxt('submission.csv', submission, delimiter=',', fmt='%d')