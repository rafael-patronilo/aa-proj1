import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

CROSS_VALIDATION_K = 5
H_RANGE=np.arange(0.02, 0.62, 0.02)


train = np.loadtxt("TP1_train.tsv", delimiter="\t")
test = np.loadtxt("TP1_test.tsv", delimiter="\t")

np.random.shuffle(train)

scaler = StandardScaler()
train[:,:-1] = scaler.fit_transform(train[:,:-1])
test[:,:-1] = scaler.transform(test[:,:-1])

kf = StratifiedKFold(n_splits=CROSS_VALIDATION_K)

class OurNB:
    """Our implementation of a Naive Bayes Classifier using Gaussian kernel. 
    Method definitions compatible with sklearn.naive_bayes.GaussianNB"""
    def __init__(self, h : float):
        self.classes = None
        self.priors = None
        self.kdes = []
        self.h = h

    def fit(self, x, y):
        self.classes = np.array(sorted(set(y)))
        self.priors = np.zeros(self.classes.shape)
        for i, cls in enumerate(self.classes):
            c_samples = (y==cls) # the indices of the samples that are in this class
            n = y[c_samples].shape[0]
            self.priors[i] = n / y.shape[0]
            kde = KernelDensity(kernel='gaussian', bandwidth=self.h)
            kde.fit(x[c_samples])
            self.kdes.append(kde)
        return self
    
    def log_priors(self):
        return np.log(self.priors)

    def predict(self, x):
        classification = np.empty((x.shape[0]))
        max_prob = np.full((x.shape[0]), -math.inf)
        cls_prob = self.log_priors()
        for i, cls in enumerate(self.classes):
            sample_prob = self.kdes[i].score_samples(x)
            cond_prob = sample_prob + cls_prob[i]
            new_max = cond_prob > max_prob
            max_prob[new_max] = cond_prob[new_max]
            classification[new_max] = cls
        return classification

    def score(self, x, y, normalize=True):
        pred = self.predict(x)
        #return accuracy_score(y, pred, normalize=normalize)
        successes = np.count_nonzero(pred == y)
        if normalize:
            return successes / y.shape[0]
        else:
            return successes

train_errors = np.empty(H_RANGE.shape)
val_errors = np.empty(H_RANGE.shape)
classifier = None
for i, h in enumerate(H_RANGE):
    train_suc = 0.0
    val_suc = 0.0
    train_tot = 0.0
    val_tot = 0.0
    for train_idx, val_idx in kf.split(train[:,:-1], train[:,[-1]]):
        x_train = train[train_idx,:-1]
        x_val = train[val_idx,:-1]
        y_train = train[train_idx,-1]
        y_val = train[val_idx,-1]
        classifier = OurNB(h)
        classifier.fit(x_train, y_train)
        train_suc += classifier.score(x_train, y_train, normalize=False)
        val_suc += classifier.score(x_val, y_val, normalize=False)
        train_tot += y_train.shape[0]
        val_tot += y_val.shape[0]
    train_errors[i] = 1.0 - train_suc / train_tot
    val_errors[i] = 1.0 - val_suc / val_tot
train_graph, = plt.plot(H_RANGE, train_errors, "b-", label="Training")
val_graph, = plt.plot(H_RANGE, val_errors, "r-", label="Validation")
best_i = val_errors.argmin()
best_h = H_RANGE[best_i]
best_point, = plt.plot(best_h, val_errors[best_i], "xg", label=f"Best: h={best_h:.2f}")
plt.xlabel("H")
plt.ylabel("Error (1-accuracy)")
plt.title("Error per H")
plt.legend(handles=[train_graph, val_graph, best_point])

our_classifier = OurNB(best_h).fit(train[:,:-1], train[:,-1])
test_error = 1.0 - our_classifier.score(test[:,:-1], test[:,-1])

print(
    f"Best H = {best_h:.2f}\n"
    "Error rates:\n"
    f"\tTraining  :  {train_errors[best_i]*100:.2f}%\n"
    f"\tValidation:  {train_errors[best_i]*100:.2f}%\n"
    f"\tTest      :  {test_error*100:.2f}%\n"
)

plt.show()
