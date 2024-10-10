import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n = 20  # 5  absolute_error: 14, squared_error: 11, friedman_mse: 12~13
criterions = ['absolute_error']  #["squared_error", "friedman_mse", "absolute_error"]  # R2: abs_err > other, MSE: abs_err < other (a little)
splitters = ["best"]  #, "random"]  # R2: random << best, MSE: random > best
# best: R2: 0.927 ~ 0.929
for i in range(1, n):
    for c in criterions:
        for s in splitters:
            tree = DecisionTreeRegressor(max_depth=i, criterion=c, splitter=s)

            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_test)

            MSE = mean_squared_error(y_test, y_pred)
            R2 = r2_score(y_test, y_pred)
            print(f"| Max Depth: {i} | criterion: {c} | splitter: {s} | R2: {R2:.3} | MSE: {MSE:.3} |")

# worse than Decision Tree
# for i in range(1, n):
#     poly = PolynomialFeatures(degree=i)
#     reg = LinearRegression()
#     X_feat = poly.fit_transform(X_train, y_train)
#     reg.fit(X_feat, y_train)
#     X_test_feat = poly.transform(X_test)
#     y_pred = reg.predict(X_test_feat)
#     MSE = mean_squared_error(y_test, y_pred)
#     R2 = r2_score(y_test, y_pred)
#     print(f"| Degree: {i} | R2: {R2:.3} | MSE: {MSE:.3} |")
