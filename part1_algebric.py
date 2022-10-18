# Not a part of the project
# 
# As a curiosity, this script attempts to predict the miss distance
# using linear algebra rather than machine learning.
import numpy as np
import matplotlib.pyplot as plt
STANDARDIZE=True
DESTANDARDIZE_TO_PLOT=STANDARDIZE

# Predict miss distance given position and velocity assuming a linear course
def predict_alg(x):
    # position
    p_r = x[:,[0]]
    p_t = x[:,[1]]
    p_n = x[:,[2]]
    # velocity
    v_r = x[:,[3]]
    v_t = x[:,[4]]
    v_n = x[:,[5]]

    # value of k in course parametic equations where course intersects the perpendicular
    #plane at (0,0,0) of the referential
    k = (v_r*p_r + v_t*p_t + v_n*p_n) / (v_r**2 + v_t**2 + v_n**2)

    # solve the parametic equations to obtain the intersection point
    i_r = p_r + v_r*k
    i_t = p_t + v_t*k
    i_n = p_n + v_n*k

    # squared miss distance
    d_sq = i_r**2 + i_t**2 + i_n**2

    return np.sqrt(d_sq)

data = np.loadtxt("SatelliteConjunctionDataRegression.csv", skiprows=1, delimiter=",")

data = data[data[:,-1].argsort()]

def mean_square_error(y, preds):
    errors = (y - preds)**2
    return np.mean(errors)

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
if STANDARDIZE:
    data = (data-mean)/std

x = data[:,:-1]
y = data[:,[-1]]

preds = predict_alg(x)

print(mean_square_error(y, preds))

if DESTANDARDIZE_TO_PLOT:
    y = y * std[-1] + mean[-1]
    preds = preds * std[-1] + mean[-1]

plt.plot(y, preds, '.b', markersize=1)
plt.plot(y, y, '--g')
plt.show()