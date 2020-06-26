import numpy as np 
import pandas as pd 

# Importing the main dataset
dataset = pd.read_csv('train.csv')

# Creating numpy arrays for independent variables X and dependent variable Y
x = dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y = dataset.iloc[:, 1].values

# Cleaning X by imputing missing values with either mean(for floats) or fill_value(for strings)
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:, 2:3] = mean_imputer.fit_transform(x[:, 2:3])
x[:, 5:6] = mean_imputer.fit_transform(x[:, 5:6])

word_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='unknown')
x[:, 6:7] = word_imputer.fit_transform(x[:, 6:7])

# OneHotEncoding categorical variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencoder = OneHotEncoder(drop='first', dtype=int)
ct = ColumnTransformer([('encoding', onehotencoder, [0, 1, 6])], remainder='passthrough')
x = ct.fit_transform(x)

# Creating training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=0)

# Scaling data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Creating, training, and testing the classifier
from test_models import test_classifier
test_classifier('Random Forest', x_train, y_train, x_test, y_test)