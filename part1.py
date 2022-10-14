import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

DEGREES = range(1, 7)

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

kf = KFold(n_splits=10)

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
    for train_idx, val_idx in kf.split(x_train):
        poly = PolynomialFeatures(degree)
        feats = poly.fit_transform(x_train)
        model = LinearRegression().fit(feats[train_idx], y_train[train_idx])
        train_error += mean_square_error(
            feats[train_idx], y_train[train_idx], model)
        val_error += mean_square_error(
            feats[val_idx], y_train[val_idx], model)
    pass

print(train_errors)
print(val_errors)

train_graph, = plt.plot(DEGREES, train_errors, label="training")
val_graph, = plt.plot(DEGREES, val_errors, label="validation")
plt.legend(handles=[train_graph, val_graph])
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.show()
