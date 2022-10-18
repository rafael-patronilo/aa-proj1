import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

DEGREES = range(1, 7)
CROSS_VALIDATION_K = 5
FORCE_WINDOW_SIZE = True
DESTANDARDIZE_TO_PLOT = True

# Calculates mean square error for
#  a given polinomial expression on a given dataset
def mean_square_error(pred, y):
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
del x_data, y_data
splits = kf.split(x_train)

poly_feats = []
train_errors = []
val_errors = []

# helper to create a straight slope 1 line and set plot limits
y_range = np.empty([2])
if DESTANDARDIZE_TO_PLOT:
    # calculate range before standardizing
    y_range[0] = y_train.min()
    y_range[1] = y_train.max()

x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

if not DESTANDARDIZE_TO_PLOT:
    # calculate range after standardizing
    y_range[0] = y_train.min()
    y_range[1] = y_train.max()

fig, axs = plt.subplots(2, len(DEGREES)//2, sharex=False, sharey=False)
fig.suptitle("Predicted to True Values")

for i, degree in enumerate(DEGREES):
    train_error = 0
    val_error = 0
    tlines=0
    vlines=0
    plot_xt = np.empty([x_train.shape[0]*(CROSS_VALIDATION_K-1),1])
    plot_xv = np.empty([x_train.shape[0],1])
    plot_yt = np.empty([x_train.shape[0]*(CROSS_VALIDATION_K-1),1])
    plot_yv = np.empty([x_train.shape[0],1])

    print(f"Degree: {degree}")

    poly = PolynomialFeatures(degree)
    feats = poly.fit_transform(x_train)
    poly_feats.append(poly)

    # Crossfold validation
    for train_idx, val_idx in kf.split(x_train): 
        model = LinearRegression().fit(feats[train_idx], y_train[train_idx])
        pred = model.predict(feats)
        train_error += mean_square_error(pred[train_idx], y_train[train_idx])
        val_error += mean_square_error(pred[val_idx], y_train[val_idx])
        # store data to plot later
        new_tlines = tlines + y_train[train_idx].shape[0]
        new_vlines = vlines + y_train[val_idx].shape[0]
        plot_xt[tlines:new_tlines] = y_train[train_idx]
        plot_xv[vlines:new_vlines] = y_train[val_idx]
        plot_yt[tlines:new_tlines] = pred[train_idx]
        plot_yv[vlines:new_vlines] = pred[val_idx]
        tlines = new_tlines
        vlines = new_vlines
    assert tlines == plot_xt.shape[0], f"plot t not full: {tlines}/{plot_xt.shape[0]}"
    assert vlines == plot_xv.shape[0], f"plot v not full: {vlines}/{plot_xv.shape[0]}"

    #Calculate average errors
    train_error /= CROSS_VALIDATION_K
    val_error /= CROSS_VALIDATION_K
    train_errors.append(train_error)
    val_errors.append(val_error)
    print(f"\ttrain: {train_error:10.4f}\tval: {val_error:10.4f}")

    if DESTANDARDIZE_TO_PLOT:
        # destandardize the data for plotting
        plot_xt = y_scaler.inverse_transform(plot_xt)
        plot_xv = y_scaler.inverse_transform(plot_xv)
        plot_yt = y_scaler.inverse_transform(plot_yt)
        plot_yv = y_scaler.inverse_transform(plot_yv)

    # plot the data
    lt, = axs.flat[i].plot(plot_xt, plot_yt, '.b', markersize=1, label=f"train: {train_error:6.2f}")
    lv, = axs.flat[i].plot(plot_xv, plot_yv, '.r', markersize=1, label=f"  val: {val_error:6.2f}")
    axs.flat[i].legend(handles=[lt, lv])
    axs.flat[i].set(xlabel = "True", ylabel = "Predicted")
    if FORCE_WINDOW_SIZE:
        axs.flat[i].set_xlim(y_range)
        axs.flat[i].set_ylim(y_range)
        axs.flat[i].label_outer()
    axs.flat[i].plot(y_range, y_range, '--g')
    axs.flat[i].set_title(f"Degree {degree}")

# plot errors to degree
plt.figure(2)
train_graph, = plt.plot(DEGREES, train_errors, "sb-", label="Training")
val_graph, = plt.plot(DEGREES, val_errors, "xr-", label="Validation")
plt.yscale("log")
plt.xlabel("Degree")
plt.ylabel("Mean Square Error")
plt.title("Error per Degree")
plt.legend(handles=[train_graph, val_graph])


# find the min validation error and the best degree
best_d = min(zip(DEGREES, val_errors, train_errors), key=lambda x : x[1:]) 
i = best_d[0] - DEGREES.start
axs.flat[i].set_title(f"Degree {best_d[0]} (best)")

# train predictor on full set with best degree
print("Obtaining best model")
feats = poly_feats[i].transform(x_train)
best_model = LinearRegression().fit(feats, y_train)

# Calculate test error
test_feats = poly_feats[i].transform(x_test)
pred = best_model.predict(test_feats)
test_error = mean_square_error(pred, y_test)

# Report final results
print(f"Best degree: {best_d[0]}")
print(f"\tValidation Error : {best_d[1]}")
print(f"\tTraining Error   : {best_d[2]}")
print(f"\tTest Error       : {test_error}")
plt.show()