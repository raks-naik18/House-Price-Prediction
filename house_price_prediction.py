import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:/JN/project/House Price India.csv")

# Data Cleaning and Preprocessing
data = data.dropna(subset=['SalePrice', 'LivArea', 'grade', 'Area_of_house(excluding_basement)', 
                           'Area_of_basement', 'number_of_bedrooms', 'YearBuilt'])

# Drop columns that won't be used as features
data = data.drop(columns=['id'])

# Separate numeric and categorical columns
numeric_cols = ['LivArea', 'grade', 'Area_of_house(excluding_basement)', 'Area_of_basement', 'number_of_bedrooms', 'YearBuilt']

# Fill missing values for numeric columns with median
imputer_numeric = SimpleImputer(strategy='median')
data[numeric_cols] = imputer_numeric.fit_transform(data[numeric_cols])

# Since there are no categorical columns mentioned, we skip the categorical processing

# Define feature matrix X and target vector y
X = data[numeric_cols]
y = data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Evaluate the Model
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f'Linear Regression - Mean Squared Error: {mse_linear}')
print(f'Linear Regression - R-squared: {r2_linear}')

# Train the Decision Tree Model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Evaluate the Model
y_pred_tree = tree_model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f'Decision Tree - Mean Squared Error: {mse_tree}')
print(f'Decision Tree - R-squared: {r2_tree}')

# Train the Random Forest Model with reduced parameters
forest_model = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10)
forest_model.fit(X_train, y_train)

# Evaluate the Model
y_pred_forest = forest_model.predict(X_test)
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print(f'Random Forest - Mean Squared Error: {mse_forest}')
print(f'Random Forest - R-squared: {r2_forest}')

# Train the Naive Bayes Model
# Note: Naive Bayes is generally used for classification problems, here we'll use it to predict SalePrice by treating it as a regression problem.
# We need to discretize the target variable for Naive Bayes
y_train_discrete = pd.cut(y_train, bins=10, labels=False)
y_test_discrete = pd.cut(y_test, bins=10, labels=False)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train_discrete)

# Evaluate the Model
y_pred_nb = nb_model.predict(X_test)
mse_nb = mean_squared_error(y_test_discrete, y_pred_nb)
r2_nb = r2_score(y_test_discrete, y_pred_nb)

print(f'Naive Bayes - Mean Squared Error: {mse_nb}')
print(f'Naive Bayes - R-squared: {r2_nb}')

# Predict new data with the trained models (example: LivArea 2000, grade 7, Area_of_house 1500, Area_of_basement 500, number_of_bedrooms 3, YearBuilt 2000)
new_data = pd.DataFrame({
    'LivArea': [2000], 
    'grade': [7], 
    'Area_of_house(excluding_basement)': [1500], 
    'Area_of_basement': [500], 
    'number_of_bedrooms': [3], 
    'YearBuilt': [2000]
})

# Reindex the new_data to match the training data columns
new_data = new_data.reindex(columns=X.columns, fill_value=0)

predicted_price_linear = linear_model.predict(new_data)
predicted_price_tree = tree_model.predict(new_data)
predicted_price_forest = forest_model.predict(new_data)

print(f'Predicted Sale Price (Linear Regression) for the example house: ${predicted_price_linear[0]:.2f}')
print(f'Predicted Sale Price (Decision Tree) for the example house: ${predicted_price_tree[0]:.2f}')
print(f'Predicted Sale Price (Random Forest) for the example house: ${predicted_price_forest[0]:.2f}')

# Plot the Results for Linear Regression
plt.scatter(y_test, y_pred_linear, color='blue', label='Linear Regression')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price (Linear Regression)')
plt.legend()
plt.show()

# Plot the Results for Decision Tree
plt.scatter(y_test, y_pred_tree, color='green', label='Decision Tree')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price (Decision Tree)')
plt.legend()
plt.show()

# Plot the Results for Random Forest
plt.scatter(y_test, y_pred_forest, color='red', label='Random Forest')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price (Random Forest)')
plt.legend()
plt.show()