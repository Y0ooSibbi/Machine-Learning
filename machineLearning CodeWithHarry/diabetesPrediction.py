# import matplotlib.pyplot as plt
# import numpy as np

# from sklearn import datasets,linear_model

# diabetes = datasets.load_diabetes()

# # ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']
# # print(diabetes.keys())
# diabetes_x = diabetes.data[:,np.newaxis,2]
# # print('helllo')
# diabetes_x_test = diabetes_x[:-30]
# diabetes_x_train = diabetes_x[-30:]


# diabetes_y_test = diabetes.target[:-30]
# diabetes_y_train = diabetes.target[-30:]

# model = linear_model.LinearRegression()
# model.fit(diabetes_x_train,diabetes_y_train)
# diabetes_y_predicted = model.predict(diabetes_x_test)
# print("Model Coef : ",model.coef_)
# print("Model Intercept : ",model.intercept_)
# print("Mean squared error is: ", mean_squared_error(diabetes_y_test,diabetes_y_predicted ))

# plt.scatter(diabetes_x_test, diabetes_y_test)
# plt.plot(diabetes_x_test, diabetes_y_predicted)

# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# plt.scatter(diabetes_X_test, diabetes_y_test)
# plt.plot(diabetes_X_test, diabetes_y_predicted)
#
# plt.show()

# Mean squared error is:  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698