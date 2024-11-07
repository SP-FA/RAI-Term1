import numpy as np
import matplotlib.pyplot as plt
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

res_dict = {
    "squared_error": [],
    "friedman_mse": [],
    "absolute_error": [],
}

n = 4  # 12 ~ 14
ne = 21
criterions = ["squared_error" , "friedman_mse", "absolute_error"]

for i in range(3, n):
    for c in criterions:
        ne_list = []
        for e in range(1, ne):
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
# plt.yticks(np.arange(0, 1, 0.05))
# plt.grid(True)
# plt.legend()
# plt.show()

plt.figure()
for c in criterions:
    res = np.array(res_dict[c])
    plt.plot(np.arange(1, ne), res[0, :, 0], label=f"criterion: {c}", alpha=0.5)
plt.xlabel("N Estimators")
plt.ylabel("R2")
plt.xticks(np.arange(1, ne))
plt.yticks(np.arange(0, 1, 0.1))
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
for c in criterions:
    res = np.array(res_dict[c])
    plt.plot(np.arange(1, ne), res[0, :, 1], label=f"criterion: {c}", alpha=0.5)
plt.xlabel("N Estimators")
plt.ylabel("MSE")
plt.xticks(np.arange(1, ne))
plt.yticks(np.arange(0, 1, 0.05))
plt.grid(True)
plt.legend()
plt.show()
