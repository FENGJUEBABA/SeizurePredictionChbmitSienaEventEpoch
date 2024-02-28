# -*- coding:utf-8 -*-
import os

import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from project1.work02.Stcnn import STCNN

# X, y = load_diabetes(return_X_y=True)
# print(type(X))
# print(type(y))
# print(X)
# print(y)
# print(X.shape)
# print(y.shape)

# features = load_diabetes()['feature_names']
# print(features)
# print(type(features))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(X_train.shape)
model = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(5,), activation='logistic', max_iter=10000, learning_rate='invscaling', random_state=0)
)
path = '../tsnedata/chb01/'
# model = STCNN()
X = []
Y = []
clas = os.listdir(path)
for cla in clas:
    if cla == '0':
        interictal_files = os.listdir(os.path.join(path, cla))
        for inter in interictal_files:
            X.append(list(np.load(os.path.join(path, cla, inter))))
            Y.append(0)
    if cla == '8':
        preictal_files = os.listdir(os.path.join(path, cla))
        for pre in preictal_files:
            X.append(list(np.load(os.path.join(path, cla, pre))))
            Y.append(1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# model = NeuralNetClassifier(model)
model.fit(X_train, y_train)
#
explainer = shap.KernelExplainer(model.predict, np.array(X_train))
#
shap_values = explainer.shap_values(np.array(X_test), nsamples=200)
#
shap.summary_plot(shap_values, X_test)
#
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], np.array(X_test)[0, :])
