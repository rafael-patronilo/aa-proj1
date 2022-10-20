import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold

CROSS_VALIDATION_K = 5
H_RANGE=np.arange(0.02, 0.62, 0.02)


train = np.loadtxt("TP1_train.tsv", delimiter="\t")
test = np.loadtxt("TP1_test.tsv", delimiter="\t")
classes = set(train[:,-1])

np.random.shuffle(train)

scaler = StandardScaler()
train[:,:-1] = scaler.fit_transform(train[:,:-1])
test[:,:-1] = scaler.transform(test[:,:-1])

kf = StratifiedKFold(n_splits=CROSS_VALIDATION_K)

def classify(kdes, feats):
    for cls, kde in kdes.items():
        kde.score_samples(feats)

    pass

for h in H_RANGE:
    for train_idx, val_idx in kf.split(train[:,:-1], train[:,[-1]]):
        train_fold = train[train_idx]
        val_fold = train[val_idx]
        kdes = {cls : KernelDensity(kernel='gaussian', bandwidth=h) for cls in classes}
        for cls, kde in kdes.items():
            cls_train = train_fold[train_fold[:,-1]==cls]
            kde.fit(cls_train)
