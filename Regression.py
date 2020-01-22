import pandas as pd
import csv
from sklearn import linear_model

# Load Train and Test datasets
train_data = pd.read_csv('population_tehran_train.csv')
test_data = pd.read_csv('population_tehran_test.csv')

x_train = train_data.drop(columns=['population'])
y_train = train_data.drop(columns=['year'])
x_test = test_data.drop(columns=['population'])

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets
linear.fit(x_train, y_train)

# Predict
predicted = linear.predict(x_test)

print("Predicted:")
for i in range (0,len(x_test)):
    print(x_test["year"][i],":",int(predicted[i]))