import json
import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

# heart_data = pd.read_csv('heart.csv')

# predictors = heart_data.drop("target",axis=1)
# target = heart_data["target"]

# X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=2)

# logReg_model = LogisticRegression()
# logReg_model.fit(X_train, Y_train)

# # Evaluate model performance
# train_accuracy = logReg_model.score(X_train, Y_train)
# test_accuracy = logReg_model.score(X_test, Y_test)

# print("Training set accuracy: {:.3f}".format(train_accuracy))
# print("Test set accuracy: {:.3f}".format(test_accuracy))

# # accuracy on training data
# X_train_prediction_lr = logReg_model.predict(X_train)
# training_data_accuracy_lr = accuracy_score(X_train_prediction_lr, Y_train)

# # accuracy on test data
# X_test_prediction_lr = logReg_model.predict(X_test)
# test_data_accuracy_lr = accuracy_score(X_test_prediction_lr, Y_test)

# # input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)
# input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)
# input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# prediction = logReg_model.predict(input_data_as_numpy_array)
# print(prediction)
# if(prediction[0]==0):
#   print('The Person does not have a Heart Disease')
# else:
#   print('The Person has Heart Disease')

# Function to load the heart dataset
def load_data():
    heart_data = pd.read_csv('heart.csv')
    predictors = heart_data.drop("target", axis=1)
    target = heart_data["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=2)
    return X_train, X_test, Y_train, Y_test

# Function to train the logistic regression model
def train_model(X_train, Y_train):
    logReg_model = LogisticRegression()
    logReg_model.fit(X_train, Y_train)
    return logReg_model

# if __name__ == "__main__":
#     # Load data
#     X_train,X_test, Y_train,Y_test = load_data()

#     # Train model
#     model = train_model(X_train, Y_train)

#     # Read input data from stdin
#     input_data = json.loads(sys.stdin.read())

#     # Make prediction
#     input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
#     prediction = model.predict(input_data_as_numpy_array)

#     # Print prediction
#     if prediction[0] == 0:
#         print('The person does not have a Heart Disease')
#     else:
#         print('The person has a Heart Disease')

if __name__ == "__main__":
    # Load data
    X_train,_ ,Y_train,_ = load_data()

    try:
        # Load trained model from file
        with open('logistic_regression_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        # Train model if file not found
        model = train_model(X_train, Y_train)
        # Save trained model to file
        with open('logistic_regression_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)

    # Read input data from stdin
    input_data = json.loads(sys.stdin.read())

    # Make prediction
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_as_numpy_array)

    # Print prediction
    if prediction[0] == 0:
        print('The person does not have a Heart Disease')
    else:
        print('The person has a Heart Disease')
