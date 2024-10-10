import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate or reuse synthetic data
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

n = 15  # 9 ~ 11
ne = 20  # 20+: similar
criterions = ["squared_error"]  # , "friedman_mse", "absolute_error"]  # R2, MSE: similar
splitters = ["best"]  # , "random"]  # similar
losses = ["linear"]  # , "square", "exponential"] # similar
# best: R2: 0.945 ~ 0.946
for i in range(1, n):
    for c in criterions:
        for s in splitters:
            for e in range(10, ne, 10):
                for l in losses:
                    bagging = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=i, criterion=c, splitter=s), n_estimators=e, random_state=42, loss=l)

                    bagging.fit(X_train, y_train)
                    y_pred = bagging.predict(X_test)

                    MSE = mean_squared_error(y_test, y_pred)
                    R2 = r2_score(y_test, y_pred)
                    print(f"| Max Depth: {i} | criterion: {c} | splitter: {s} | n_estimators: {e} | loss: {l} | R2: {R2:.3} | MSE: {MSE:.3} |")