import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

# Define the dataset and hyperparameter space
simulate_data_A = pd.read_csv('C:/Users/rmaat\Desktop/MSc/data_A/data_scenario_G_n_500_dataset_0.csv')
X = simulate_data_A.drop(columns=['T', 'Y'])
T = simulate_data_A['T']
Y = simulate_data_A['Y']
W = 2 * T - 1

# Define the hyperparameter space
dim_C = Real(low=1e-6, high=1e+6, prior='log-uniform', name='C')
dim_kernel = Categorical(categories=['linear', 'rbf', 'poly'], name='kernel')
dim_gamma = Real(low=1e-6, high=1e+1, prior='log-uniform', name='gamma')
dim_degree = Real(low=1, high=5, name='degree')  # Use Int space for degree

dimensions = [dim_C, dim_kernel, dim_gamma, dim_degree]

# Define the target score function
@use_named_args(dimensions=dimensions)
def objective(**params):
    params['degree'] = round(params['degree'])  # Round degree to the nearest integer
    clf = SVC(**params)
    X_train, X_test, y_train, y_test = train_test_split(X, W, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return -score  # We negate because we want to maximize accuracy

# Initialize the Bayesian Optimization
n_initial_points = 5
n_calls = 20

result = gp_minimize(objective, dimensions, n_initial_points=n_initial_points, n_calls=n_calls, random_state=42, verbose=True)

# Extract optimized hyperparameters
best_params = result.x
best_score = -result.fun  # Negate to get the actual score

print("Best Hyperparameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# Build the final SVM model using the optimized hyperparameters
final_clf = SVC(C=best_params[0], kernel=best_params[1], gamma=best_params[2], degree=best_params[3])
final_clf.fit(X, W)

# Evaluate the final model on the test set (for simplicity, we use the same dataset here)
test_score = final_clf.score(X, W)
print("Test Set Accuracy:", test_score)