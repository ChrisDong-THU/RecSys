import torch
from sklearn.metrics import log_loss, roc_auc_score


from deepctr_torch.models import *

from utils import read_data


def train():
    train_data, test_data, info = read_data('train.txt', 5000000)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = DeepFM(linear_feature_columns=info[1], dnn_feature_columns=info[0], \
        task='binary', l2_reg_linear=1e-2, l2_reg_embedding=1e-2, dnn_dropout=1e-2, device=device)
    
    model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )

    history = model.fit(train_data[0], train_data[1], batch_size=64, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(test_data[0], 256)
    print("test LogLoss", round(log_loss(test_data[1], pred_ans), 4))
    print("test AUC", round(roc_auc_score(test_data[1], pred_ans), 4))
    

if __name__ == '__main__':
    train()