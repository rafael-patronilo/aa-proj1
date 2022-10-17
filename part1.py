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
np.random.shuffle(data)

x_data = data[:, :-1]  # position 3d vector, velocity 3d vector
y_data = data[:, [-1]]  # miss distance

kf = KFold(n_splits=CROSS_VALIDATION_K)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2)
splits = kf.split(x_train)

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
    print(f"Degree: {degree}")
    train_error = 0
    val_error = 0
    poly = PolynomialFeatures(degree)
    feats = poly.fit_transform(x_train)
    poly_feats.append(poly)
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
    print(f"\ttrain: {train_error:10.4f}\tval: {val_error:10.4f}")

# find the min validation error and the best degree
best_d = min(zip(DEGREES, val_errors, train_errors), key=lambda x : x[1:]) 

# plot errors to degree
plt.figure(1)
train_graph, = plt.plot(DEGREES, train_errors, "sb-", label="Training")
val_graph, = plt.plot(DEGREES, val_errors, "xr-", label="Validation")
plt.yscale("log")
plt.xlabel("Degree")
plt.ylabel("Mean Square Error")
plt.title("Error per Degree")
plt.legend(handles=[train_graph, val_graph])

# Train models on the full training set
models = []

for i, _ in enumerate(DEGREES):
    poly = poly_feats[i]
    feats = poly.transform(x_train)
    model = LinearRegression().fit(feats, y_train)
    models.append(model)


# For each degree plot predicted to true values
x_eval = x_train #np.vstack((x_train,x_test))
y_eval = y_train #np.vstack((y_train,y_test))
fig, axs = plt.subplots(2, 3, sharex = True, sharey = True)
fig.suptitle("Predicted to True Values")
for i, degree in enumerate(DEGREES):
    poly = poly_feats[i]
    feats = poly.transform(x_eval)

    pred = models[i].predict(feats)
    # order = np.argsort(pred[:,0])
    axs.flat[i].plot(y_eval, pred,'.', markersize=1)
    axs.flat[i].set(xlabel = "True", ylabel = "Predicted")
    axs.flat[i].plot(y_eval, y_eval)
    axs.flat[i].set_title(f"Degree {degree}" + (" (best)" if degree == best_d[0] else ""))

plt.show()


best_model = models[best_d[0] + DEGREES.start]

test_feats = poly_feats[best_d[0] + DEGREES.start].transform(x_test)
test_error = mean_square_error(test_feats, y_test, best_model)

print(f"Best degree: {best_d[0]}")
print(f"\tValidation Error : {best_d[1]}")
print(f"\tTraining Error   : {best_d[2]}")
print(f"\tTest Error       : {test_error}")

