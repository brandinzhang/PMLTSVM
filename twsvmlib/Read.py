import pandas as pd





# 定义每个数据集的标签列索引范围（这里需要根据实际情况修改）
label_cols = {
    'birds': slice(-19, None),  # birds 数据集最后 19 列是标签列
    'emotions': slice(-6, None),  # emotions 数据集最后 6 列是标签列
    'flags': slice(-7, None),  # flags 数据集最后 7 列是标签列
    'yeast': slice(-14, None),  # yeast 数据集最后 14 列是标签列
    'genbase': slice(-27, None),  # genbase 数据集最后 27 列是标签列
    'scene': slice(-6, None),  # scene 数据集最后 6 列是标签列   
    'CAL500': slice(-174, None),  # CAL500 数据集最后 20 列是标签列
    'medical':slice(-45,None) # medical 数据集最后 45 列是标签列
}



def read(name):
    # 读取数据集
    path = 'dataset/' + name + '.csv'
    data = pd.read_csv(path, header=None)
    X = data.iloc[:, :label_cols[name].start].values
    Y = data.iloc[:, label_cols[name]].values
    return X,Y






"""
# 读取数据集,打印详细
X,y = read('birds')
print(y[1])
"""













