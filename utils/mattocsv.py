import scipy.io
import pandas as pd
import numpy as np


mat_data = scipy.io.loadmat('image.mat')

features = mat_data['data']  # (2000, 特征维度)
labels = mat_data['target']  # (2000, 5)




if features.shape[0] != 2000:
    features = features.T  

if labels.shape[0] != 2000:
    labels = labels.T


feature_columns = [f'feat_{i}' for i in range(features.shape[1])]
label_columns = [f'label_{i}' for i in range(5)]

df = pd.DataFrame(features, columns=feature_columns)
df[label_columns] = labels  


df.to_csv('image.csv', index=False)