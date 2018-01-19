#!/usr/bin/env python

import numpy as np
import pickle
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV


# load the data matrix
file_pkl = open("face.pkl", "rb")
face = pickle.load(file_pkl)
file_pkl.close()

# run PCA on the original 960 dimensional feature vector
X_raw = face['X']
pca = PCA(n_components = 20)
X = scale(pca.fit_transform(X_raw))
print "explained variance ratio = " + str(pca.explained_variance_ratio_.sum())

# extract two alternative cluster labels
Y_identity = face['Y_identity']
Y_pose = face['Y_pose']

# fit multiple logistic regression
clf = LogisticRegressionCV(cv=5, max_iter=500, n_jobs=-1,
        multi_class='multinomial', random_state=0)

#clf.fit(X, Y_pose)
clf.fit(X, Y_identity)
print Y_pose
print "Accuracy of LR = " + str(clf.scores_[0].mean(axis=0).max())


N = X.shape[0]
d = X.shape[1]
L = str(N) + '_' + str(d)
np.savetxt('face_' + L + '.csv', X, delimiter=',', fmt='%f')
np.savetxt('face_' + L + '_original_label.csv', Y_identity, delimiter=',', fmt='%f')
np.savetxt('face_' + L + '_alt_label.csv', Y_pose, delimiter=',', fmt='%f')

