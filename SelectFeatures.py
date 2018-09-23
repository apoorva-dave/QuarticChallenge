import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedShuffleSplit

# reading data
data = pd.read_csv("./ds_data/data_train.csv")
print(data.head())

# print(data.describe())

data = data.fillna(0)
print("length of 0s data",len(data[data["target"]==0]))
print("length of 1s  data",len(data[data["target"]==1]))

# printing columns of data
print(data.columns)

features = list(data.columns[1:57])
print(features)

label = list(data.columns[57:])
print(label)

X = data[features].values
print(X.shape)
y = data[label].values
y = y.ravel()

df = pd.DataFrame()
df['features'] = features
# print(df)
#
# print(df.columns.values)
# print(df.features[0])
print('Selecting 30 best features')
selector = SelectKBest(chi2, k=30)
X = selector.fit_transform(X,y)
idx_selected = selector.get_support(indices=True)
print(idx_selected)
new_features = df.features[idx_selected]
print('Selected features')
print(new_features)

print(X.shape)

# # stratfied sampling of data
# sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
# sss.get_n_splits(X, y)
#
# print(sss)
#
# for train_index, test_index in sss.split(X, y):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

# # the entire data is training data set while creation of final model
