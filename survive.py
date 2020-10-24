import numpy as np 
import pandas as pd 

# =========================================TRAINING======================================
# Importing the training dataset
train = pd.read_csv('data/train.csv')

# Creating family_size array by adding SibSp and Parch
train_family_size = train['SibSp'] + train['Parch'] + 1
train_family_size = np.array(train_family_size, dtype=int)
train_family_size = train_family_size.reshape(len(train_family_size), 1)

# Creating numpy arrays for independent variables x_train and dependent variable y_train
x_train = train.iloc[:, [2, 4, 5, 9, 11]].values
y_train = train.iloc[:, 1].values

# Concatenating x_train and train_family_size
x_train = np.concatenate((x_train, train_family_size), axis=1)

# Cleaning x_train by imputing missing values with either mean (for floats) or most frequent (for strings)
from sklearn.impute import SimpleImputer
mean_age = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train[:, 2:3] = mean_age.fit_transform(x_train[:, 2:3])

mean_fare = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train[:, 3:4] = mean_fare.fit_transform(x_train[:, 3:4])

word_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x_train[:, 4:5] = word_imputer.fit_transform(x_train[:, 4:5])

# OneHotEncoding categorical variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencoder = OneHotEncoder(drop='first', dtype=int)
ct = ColumnTransformer([('encoding', onehotencoder, [0, 1, 4])], remainder='passthrough')
x_train = ct.fit_transform(x_train)

# Scaling data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)

# Creating and training the classifier
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

# =========================================TESTING======================================
# Importing the testing dataset
test = pd.read_csv('data/test.csv')

# Creating family_size array by adding SibSp and Parch
test_family_size = test['SibSp'] + test['Parch'] + 1
test_family_size = np.array(test_family_size, dtype=int)
test_family_size = test_family_size.reshape(len(test_family_size), 1)

# Creating numpy array for the necessary testing columns
x_test = test.iloc[:, [1, 3, 4, 8, 10]].values
ids = test.iloc[:, 0].values

# Cocatenating x_train and test_family_size
x_test = np.concatenate((x_test, test_family_size), axis=1)

# Cleaning x_test by imputing missing values with mean(for floats)
x_test[:, 2:3] = mean_age.transform(x_test[:, 2:3])
x_test[:, 3:4] = mean_fare.transform(x_test[:, 3:4])

# OneHotEncoding categorical variables
x_test = ct.transform(x_test)

# Scaling data
x_test = sc_x.transform(x_test)

# Predicting with the classifier
predictions = classifier.predict(x_test)

# Reshaping ids and predictions for concatenation
ids = ids.reshape(len(ids), 1)
predictions = predictions.reshape(len(predictions), 1)

# Cocatenating passenger ids and predictions
submission = np.concatenate((ids, predictions), axis=1)

# Saving submission to submission.csv
np.savetxt('submission.csv', submission, delimiter=',', fmt='%d')
