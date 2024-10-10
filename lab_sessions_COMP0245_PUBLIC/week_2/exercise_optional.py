import numpy as np
from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n = 25  # 12 ~ 14
ne = 50
criterion = "squared_error"
bagging = RandomForestRegressor(max_depth=n, criterion=criterion, n_estimators=ne, random_state=42)

bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
print(f"| R2: {R2:.3} | MSE: {MSE:.3} |")
