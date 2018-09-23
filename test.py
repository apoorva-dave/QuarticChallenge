import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import SelectFeatures

data2 = pd.read_csv("./ds_data/data_test.csv")
print(data2.head())
data2 = data2.fillna(0)

new_features = SelectFeatures.df.features[SelectFeatures.idx_selected]
print(new_features)
features = list(new_features)
print(features)
test_X = data2[features].values
print(test_X.shape)



from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
test_X = scaler.fit_transform(test_X)
print(test_X)

pca = PCA(0.95)
print(pca.n_components)
pca.fit(test_X)
test_X = pca.transform(test_X)
print(test_X.shape)

df = pd.DataFrame(columns = ['id','target'])
df.id = data2.id
print("Loading model")
filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
print("testing test data")
y_pred = loaded_model.predict(test_X)
print(y_pred)
print(len(y_pred))
df.target = y_pred
# print(df.target)
print(df.head())

print("saving result")
df.to_csv('result.csv', encoding='utf-8', index=False)
print(df.head())