# Not a part of the project
# 
# As a curiosity, this script attempts to predict the miss distance
# using linear algebra rather than machine learning.
import numpy as np
import matplotlib.pyplot as plt

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
    # standardize before calculating the error, in order for it to be comparable
    mean = np.mean(y)
    std = np.std(y)
    y = (y-mean)/std
    preds = (preds-mean)/std
    errors = (y - preds)**2
    return np.mean(errors)

x = data[:,:-1]
y = data[:,[-1]]

preds = predict_alg(x)

print(mean_square_error(y, preds))

plt.plot(y, preds)
plt.show()