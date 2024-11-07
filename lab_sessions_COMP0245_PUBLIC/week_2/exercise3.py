import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

res_dict = {
    "squared_error": [],
    "friedman_mse": [],
    "absolute_error": [],
}

n = 6  # 12~14
ne = 51  # 30+: similar
criterions = ["squared_error" , "friedman_mse", "absolute_error"]  # R2, MSE: similar, absolute_error: worst
# best: R2: 0.95 ~ 0.952
for i in range(5, n):
    for c in criterions:
        ne_list = []
        for e in range(10, ne, 10):
            bagging = RandomForestRegressor(max_depth=i, criterion=c, n_estimators=e, random_state=42)

            bagging.fit(X_train, y_train)
            y_pred = bagging.predict(X_test)

            MSE = mean_squared_error(y_test, y_pred)
            R2 = r2_score(y_test, y_pred)
            print(f"| Max Depth: {i} | criterion: {c} | n_estimators: {e} | R2: {R2:.3} | MSE: {MSE:.3} |")
            ne_list.append([R2, MSE])
        res_dict[c].append(ne_list)

# plt.figure()
# for c in criterions:
#     res = np.array(res_dict[c])
#     plt.plot(res[:, 0, 0], label=f"criterion: {c}", alpha=0.5)
# plt.xlabel("Max Depth")
# plt.ylabel("R2")
# plt.xticks(np.arange(0, 20, 1))
# plt.yticks(np.arange(0, 1, 0.1))
# plt.grid(True)
# plt.legend()
# plt.show()
#
# plt.figure()
# for c in criterions:
#     res = np.array(res_dict[c])
#     plt.plot(res[:, 0, 1], label=f"criterion: {c}", alpha=0.5)
# plt.xlabel("Max Depth")
# plt.ylabel("MSE")
# plt.xticks(np.arange(0, 20, 1))
# plt.yticks(np.arange(0, 0.3, 0.05))
# plt.grid(True)
# plt.legend()
# plt.show()

plt.figure()
for c in criterions:
    res = np.array(res_dict[c])
    plt.plot(np.arange(10, 51, 10), res[0, :, 0], label=f"criterion: {c}", alpha=0.5)
plt.xlabel("N Estimators")
plt.ylabel("R2")
plt.xticks(np.arange(10, 51, 10))
plt.yticks(np.arange(0, 1, 0.1))
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
for c in criterions:
    res = np.array(res_dict[c])
    plt.plot(np.arange(10, 51, 10), res[0, :, 1], label=f"criterion: {c}", alpha=0.5)
plt.xlabel("N Estimators")
plt.ylabel("MSE")
plt.xticks(np.arange(10, 51, 10))
plt.yticks(np.arange(0, 0.3, 0.05))
plt.grid(True)
plt.legend()
plt.show()
