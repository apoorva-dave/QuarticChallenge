from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import grid_search
import SelectFeatures


# scaling train data
scaler  = StandardScaler()
X_train = scaler.fit_transform(SelectFeatures.X)
# X_test = scaler.fit_transform(SelectFeatures.X_test)

# applying pca
pca = PCA(0.95)
print(pca.n_components)
pca.fit(X_train)
X_train = pca.transform(X_train)
print(X_train.shape)
# X_test = pca.transform(X_test)

print("train data created")
print("training model..")



parameters = {'C':[1, 10]}
clf = linear_model.LogisticRegression(class_weight = 'balanced')
clf = grid_search.GridSearchCV(clf, parameters)
clf = clf.fit(X_train,SelectFeatures.y)
# y_pred = clf.predict(X_test)

import pickle
print("model trained.saving model..")
filename = 'model.sav'
pickle.dump(clf, open(filename, 'wb'))
print("model saved")

# accuracy measure - AUC and confusion matrix


# from sklearn import metrics
# fpr, tpr, thresholds = metrics.roc_curve(SelectFeatures.y_test, y_pred)
# print(metrics.auc(fpr, tpr))            #0.58
# #
# from sklearn.metrics import confusion_matrix,classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# cnf_matrix = confusion_matrix(SelectFeatures.y_test, y_pred)
# print("the recall for this model is :", cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0]))
# fig = plt.figure(figsize=(6, 3))
# print("TP", cnf_matrix[1, 1,])
# print("TN", cnf_matrix[0, 0])
# print("FP", cnf_matrix[0, 1])
# print("FN", cnf_matrix[1, 0])
# sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
# plt.title("Confusion_matrix")
# plt.xlabel("Predicted_class")
# plt.ylabel("Real class")
# plt.show()
# print("\n----------Classification Report------------------------------------")
# print(classification_report(SelectFeatures.y_test, y_pred))
