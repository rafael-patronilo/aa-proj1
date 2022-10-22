import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

CROSS_VALIDATION_K = 5
H_RANGE=np.arange(0.02, 0.62, 0.02)
CONFIDENCE=0.95
MC_NEMAR_CHI_SQ=3.84 #95% confidence


train = np.loadtxt("TP1_train.tsv", delimiter="\t")
test = np.loadtxt("TP1_test.tsv", delimiter="\t")

np.random.shuffle(train)

scaler = StandardScaler()
train[:,:-1] = scaler.fit_transform(train[:,:-1])
test[:,:-1] = scaler.transform(test[:,:-1])

kf = StratifiedKFold(n_splits=CROSS_VALIDATION_K)

def successes(classifier, x, y):
    pred = classifier.predict(x)
    return np.count_nonzero(pred == y)

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

    def score(self, x, y):
        succ = successes(self, x, y)
        return succ / y.shape[0]

train_errors = np.empty(H_RANGE.shape)
val_errors = np.empty(H_RANGE.shape)
for i, h in enumerate(H_RANGE):
    train_acc = 0.0
    val_acc = 0.0
    for train_idx, val_idx in kf.split(train[:,:-1], train[:,[-1]]):
        x_train = train[train_idx,:-1]
        x_val = train[val_idx,:-1]
        y_train = train[train_idx,-1]
        y_val = train[val_idx,-1]
        classifier = OurNB(h)
        classifier.fit(x_train, y_train)
        train_acc += classifier.score(x_train, y_train)
        val_acc += classifier.score(x_val, y_val)
    train_errors[i] = 1.0 - train_acc / CROSS_VALIDATION_K
    val_errors[i] = 1.0 - val_acc / CROSS_VALIDATION_K
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
    f"\tTest      :  {test_error*100:.2f}%"
)

skl_classifier = GaussianNB().fit(train[:,:-1], train[:,-1])
skl_error = 1.0 - skl_classifier.score(test[:,:-1], test[:,-1])
print(f"sklearn GaussianNB test error rate: {skl_error*100:.2f}%")
print()

def normal_interval(classifier, x, y):
    """Obtains the approximate normal interval for the true error

    Args:
        classifier : the classifier to test
        x : test features, shape (n_samples, n_features)
        y : test classes, shape (n_samples,)

    Returns:
        (errors, difference): variables representing
        the expected interval for the true error.
        The true error is expected to be between errors - difference and
        errors + difference
    """
    n = x.shape[0]
    s = successes(classifier, x, y)
    errors = n - s
    dev = math.sqrt(n * (errors/n) * (1 - errors/n))
    return errors, dev * (1.01 + CONFIDENCE)

def normal_test(classifier1, classifier2, x, y):
    """Compare both classifiers using approximate normal test

    Args:
        classifier1: The first classifier to test
        classifier2: The second classifier to test
        x: test features, shape (n_samples, n_features)
        y: test classes, shape (n_samples,)

    Returns:
        compare: a positive value if classifier 1 is better and a 
        negative value if classifier 2 is better. None if the test
        could not determine which is better
        normal_test1: the interval for the true error of classifier 1
        normal_test2: the interval for the true error of classifier 2
    """
    normal_test1 = normal_interval(classifier1, x, y)
    normal_test2 = normal_interval(classifier2, x, y)
    compare = None
    if normal_test1[0] + normal_test1[1] < normal_test2[0] - normal_test2[1]:
        # 1 is better
        compare = 1
    elif normal_test2[0] + normal_test2[1] < normal_test1[0] - normal_test1[1]:
        # 2 is better
        compare = -1
    #else: # impossible to compare
    return compare, normal_test1, normal_test2
        

def mc_nemar_test(classifier1, classifier2, x, y):
    """_summary_

    Args:
        classifier1: The first classifier to test
        classifier2: The second classifier to test
        x: test features, shape (n_samples, n_features)
        y: test classes, shape (n_samples,)

    Returns:
        compare: a positive value if classifier 1 is better and a 
        negative value if classifier 2 is better. None if the test
        could not determine which is better
        significance: the value that was obtained to determine if there
        is a significant difference between both classifiers
        ex1: the number of errors in the predictions of classifier 1 but not classifier 2
        ex2: the number of errors in the predictions of classifier 2 but not classifier 1
    """
    errors1 = classifier1.predict(x) != y
    errors2 = classifier2.predict(x) != y
    # samples that aren't wrong in both classifiers
    mask = ~(errors1 & errors2)
    # count errors exclusive to each classifier
    ex1 = np.count_nonzero(errors1 & mask)
    ex2 = np.count_nonzero(errors2 & mask)
    significance = (
        (abs(ex1 - ex2) - 1)**2 /
        (ex1 + ex2)
    )
    significant = significance >= MC_NEMAR_CHI_SQ
    compare = -(ex1 - ex2) if significant else None
    return compare, significance, ex1, ex2

normal_compare, our_interval, skl_interval = normal_test(our_classifier, skl_classifier, test[:,:-1], test[:,-1])
mc_nemar_compare, sig, our_errors, skl_errors = mc_nemar_test(our_classifier, skl_classifier, test[:,:-1], test[:,-1])



def conclude(compare_value):
    if compare_value == None:
        return "undetermined"
    elif compare_value > 0:
        return f"OurNB({best_h:.2f})"
    elif compare_value < 0:
        return f"GaussianNb()"
    else:
        return "equivalent"


print(
    "Approximate Normal Test Result:\n"
    f"\tOurNB({best_h:.2f})  : {our_interval[0]:4} ± {our_interval[1]:4.2f}\n"
    f"\tGaussianNB() : {skl_interval[0]:4} ± {skl_interval[1]:4.2f}\n"
    f"Best classifier: {conclude(normal_compare)}" 
)
print()

print(
    f"McNemar Test Result: {sig:.4f}\n"
    "Number of errors in the predictions of exclusively each classifier\n"
    f"\tOurNB({best_h:.2f})  : {our_errors:4}\n"
    f"\tGaussianNB() : {skl_errors:4}\n"
    f"Best classifier: {conclude(mc_nemar_compare)}" 
)

plt.show()
