from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR  # For regression instead of SVC for classification
from sklearn.linear_model import LinearRegression  # For regression
from sklearn.tree import DecisionTreeRegressor  # For regression
from sklearn.ensemble import RandomForestRegressor  # For regression
from sklearn import datasets
from sklearn.metrics import mean_absolute_error

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Naïve Bayes Regressor
nb_regressor = GaussianNB()
nb_regressor.fit(X_train, y_train)
nb_predictions = nb_regressor.predict(X_test)
nb_mae = mean_absolute_error(y_test, nb_predictions)
print(f"Naïve Bayes Regressor MAE: {nb_mae}")

# Support Vector Machine (SVM) Regressor
svm_regressor = SVR()
svm_regressor.fit(X_train, y_train)
svm_predictions = svm_regressor.predict(X_test)
svm_mae = mean_absolute_error(y_test, svm_predictions)
print(f"SVM Regressor MAE: {svm_mae}")

# Linear Regression
lr_regressor = LinearRegression()
lr_regressor.fit(X_train, y_train)
lr_predictions = lr_regressor.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)
print(f"Linear Regression MAE: {lr_mae}")

# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)
dt_predictions = dt_regressor.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_predictions)
print(f"Decision Tree Regressor MAE: {dt_mae}")

# Random Forest Regressor
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print(f"Random Forest Regressor MAE: {rf_mae}")
