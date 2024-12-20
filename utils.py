import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names


root_path = 'E:/VSCodeProj/Dataset/criteo'
NAMES = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
         'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
         'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
         'C23', 'C24', 'C25', 'C26'] # test.txt没有label列
TARGET = 'label'


def read_data(file, num):
    file = os.path.join(root_path, file)
    data = pd.read_csv(file, sep='\t', iterator=True, names=NAMES)
    data = data.get_chunk(num)
    
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    # 密集特征填充0，稀疏特征填充-1
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 标签编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 计算每个特征的唯一值个数
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4) for feat in sparse_features] \
        + [DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    train, test = train_test_split(data, test_size=0.2, random_state=2024)
    
    train_input = {name: train[name] for name in feature_names}
    test_input = {name: test[name] for name in feature_names}
    train_target = train[TARGET].values
    test_target = test[TARGET].values
    

    return (train_input, train_target), (test_input, test_target), (dnn_feature_columns, linear_feature_columns)


if __name__ == '__main__':
    data, feature_names = read_data('test.txt', 100)
    pass
