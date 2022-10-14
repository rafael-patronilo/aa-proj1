import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

DEGREES = range(1, 7)
CROSS_VALIDATION_K = 5

# Calculates mean square error for
#  a given polinomial expression on a given dataset


def mean_square_error(x, y, reg):
    pred = reg.predict(x)
    error = np.mean((pred-y)**2)
    return error


data = np.loadtxt("SatelliteConjunctionDataRegression.csv",
                  skiprows=1, delimiter=",")

# np.random.seed(42)

x_data = data[:, :-1]  # position 3d vector, velocity 3d vector
y_data = data[:, [-1]]  # miss distance

kf = KFold(n_splits=CROSS_VALIDATION_K)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2)
splits = kf.split(x_train)

models = []
poly_feats = []
train_errors = []
val_errors = []

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

for degree in DEGREES:
    train_error = 0
    val_error = 0
    poly = PolynomialFeatures(degree)
    feats = poly.fit_transform(x_train)
    for train_idx, val_idx in kf.split(x_train):    
        model = LinearRegression().fit(feats[train_idx], y_train[train_idx])
        train_error += mean_square_error(
            feats[train_idx], y_train[train_idx], model)
        val_error += mean_square_error(
            feats[val_idx], y_train[val_idx], model)
    train_error /= CROSS_VALIDATION_K
    val_error /= CROSS_VALIDATION_K
    train_errors.append(train_error)
    val_errors.append(val_error)

best_d = min(zip(DEGREES, val_errors, train_errors), key=lambda x : x[1:]) 
print(f"Best degree: {best_d[0]}")
print(f"\t Validation Error: {best_d[1]}")
print(f"\t Training Error  : {best_d[2]}")

train_graph, = plt.plot(DEGREES, train_errors, label="training")
val_graph, = plt.plot(DEGREES, val_errors, label="validation")
plt.yscale("log")
plt.xlabel("Degree")
plt.ylabel("mean_square_error")
plt.legend(handles=[train_graph, val_graph])
plt.show()
