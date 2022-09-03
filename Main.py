import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import torch.nn as nn
import numpy as np

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # The number of output features is equal to the number of input planes.
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MLSTMfcn(nn.Module):
    def __init__(self, *, num_classes, max_seq_len, num_features,
                 num_lstm_out=128, num_lstm_layers=1,
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.8, fc_drop_p=0.3):
        super(MLSTMfcn, self).__init__()

        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)  # ex 128   # AdaptiveAvgPool1d
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes)

    def forward(self, x, seq_lens):
        ''' input x should be in size [B,T,F], where
            B = Batch size
            T = Time samples
            F = features
        '''
        # print("x", x.shape) # ([2, 244, 28])
        x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens,
                                               batch_first=True,
                                               enforce_sorted=False)  # 先填充后压缩
        # print("x1", x1.data.shape) #  x1 torch.Size([变长, 28])
        x1, (ht, ct) = self.lstm(x1)
        x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True,
                                                 padding_value=0.0)  # 压缩后填充
        x1 = x1[:, -1, :]

        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)  # AdaptiveAvgPool1d
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out

def validation(model, testloader, criterion, device='cpu'):
    accuracy = 0
    test_loss = 0
    res = None
    for inputs, labels, seq_lens in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs, seq_lens)
        # test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        # equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        # accuracy += equality.type_as(torch.FloatTensor()).mean()

        res = ps.max(1)[1]
    return res

    # return test_loss, accuracy

# def load_EV_datasets(folder, max_seq_len, dataset_name='EV'):
#     seq_lens_train = [] # 记录每个sample的 有效值，len的长度
#     X_train = []
#     file_list = sorted(os.listdir(folder)) # 要sort， 和 y_train 对应。
#     features = ["V_Cell_max", "V_Cell_min", "T_max", "T_min", "SOC"]
#     # y_train = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] # typeB
#     # y_train = [0, 0, 1, 1, 1, 1, 0, 1, 1, 1]   # typeA: ['01', '02', '03', '04', '05', '06', '11', '12', '13', '14']
#     y_train = [0]
#     num_samples = len(y_train)
#
#     print("features:", len(features))
#     print("file_list", file_list)
#
#     # 增加一个 temp_feature 使其 达到长度，用于pad_sequence,
#     temp_feature = torch.Tensor(np.array([0 for _ in range(max_seq_len)]))
#     X_train.append(temp_feature)
#
#     # 加载数据， 产生 X_train
#     for file in file_list[:num_samples]:
#         data = pd.read_csv(folder + file)
#         seq_lens_train.append(len(data))
#         for feature in features:
#             # print(data[feature])
#             # print(type(data[feature]))
#             X_train.append(torch.Tensor(data[feature]))
#
#     # 填充
#     X_train = pad_sequence(X_train, batch_first=True)
#     print("X_train", X_train.shape)
#
#     # delete the temp_feature
#     X_train = X_train[1:]
#     print("X_train", X_train.shape)
#
#     # 将第0维度 分割开。 (samples * feature, length) --> (samples, feature, length)
#     X_train = torch.chunk(X_train, chunks=num_samples, dim=0)  # chunk 后 成为tuple 了 # (tensor, tensor,...)
#     # print("X_train", X_train[0])
#
#     # 格式转化 (tensor, tensor,...) --> [numpy, numpy,...] -->  tensor[]
#     X_train_tensor = []
#     for i in X_train:
#         X_train_tensor.append(i.numpy())
#     X_train_tensor = torch.Tensor(X_train_tensor)
#     # print("X_train_tensor", X_train_tensor)
#     print("X_train_tensor", X_train_tensor.shape)
#
#     # 交换维度 permute
#     X_train = X_train_tensor.permute(0, 2, 1)
#     print("X_train", X_train.shape)
#
#     y_train = torch.tensor(y_train, dtype=torch.int64)
#     print("y_train", y_train)
#
#     seq_lens_train = torch.tensor(seq_lens_train, dtype=torch.int64)
#     print("seq_lens_train",seq_lens_train)
#
#     # todo 改正数据集。
#     # X_val, y_val, seq_lens_val = X_train, y_train, seq_lens_train
#     X_test, y_test, seq_lens_test = X_train, y_train, seq_lens_train
#
#     # train_dataset = torch.utils.data.TensorDataset(X_train, y_train, seq_lens_train)
#     # val_dataset = torch.utils.data.TensorDataset(X_val, y_val, seq_lens_val)
#
#     test_dataset = torch.utils.data.TensorDataset(X_test, y_test, seq_lens_test)
#
#     return test_dataset

def clean(path_read, max_seq_len):
    '''
    针对每一辆车：
    合并 文件based on time -- 选取features -- drop nan -- 存入 temp_data 文件夹中
    '''
    car_data_file_list = sorted(os.listdir(path_read))
    if car_data_file_list[0] == '.DS_Store':
        car_data_file_list = car_data_file_list[1:]
    print(car_data_file_list)

    # 7个data文件合起来
    car_dataframes = []
    for i in range(len(car_data_file_list)):
        car_dataframes.append(pd.read_csv(path_read + car_data_file_list[i]))
    car = pd.concat(car_dataframes)


    # 选取 features
    useful_features = ['time', 'V_Cell_max', 'V_Cell_min', 'T_max', 'T_min', 'SOC']
    car = pd.DataFrame(car, columns=useful_features)

    # clean data drop nan
    car = car.dropna(axis=0, how="any")


    # path_out = "./temp/car.csv"
    # car.to_csv(path_out)
    # print("car",car)

    # todo sampling
    # car = pd.read_csv(path_out)
    # print("len(car)", len(car))

    if len(car) <= 200 * max_seq_len:
        rate = 200
        data = car[::rate]
    else:
        rate = 2000
        data = car[::rate]
    # print("data", data)
    if len(car) > 200 * max_seq_len:
        data = car[::200]
    return data

def load_EV_datasets(data, max_seq_len, car_Type):
    seq_lens_train = [] # 记录每个sample的 有效值，len的长度
    X_train = []

    # 增加一个 temp_feature 使其 达到长度，用于pad_sequence,
    temp_feature = torch.Tensor(np.array([0 for _ in range(max_seq_len)]))
    X_train.append(temp_feature)

    features = ["V_Cell_max", "V_Cell_min", "T_max", "T_min", "SOC"]

    # 加载3个临时 数据， 产生 X_train
    car_A_name = ["car_01.csv", "car_02.csv","car_03.csv","car_04.csv","car_05.csv","car_06.csv",
                  "car_11.csv","car_12.csv","car_13.csv","car_14.csv",]

    car_B_name = ["car_07.csv", "car_08.csv","car_09.csv","car_10.csv","car_15.csv","car_16.csv",
                  "car_17.csv","car_18.csv","car_19.csv","car_20.csv","car_21.csv","car_22.csv",
                  "car_23.csv","car_24.csv","car_25.csv","car_26.csv","car_27.csv","car_28.csv",
                  "car_29.csv","car_30.csv",]

    data_temp = []
    if car_Type == "A":
        for A in car_A_name:
            d_temp = pd.read_csv("./temp_data2/typeA/" + A)
            data_temp.append(d_temp)
        y_train = [0 for _ in range(len(car_A_name) + 1)]
    if car_Type == "B":
        for B in car_B_name:
            d_temp = pd.read_csv("./temp_data2/typeB/" + B)
            data_temp.append(d_temp)
        y_train = [0 for _ in range(len(car_B_name) + 1)]

    print("y_train", y_train)

    for i in range(len(data_temp)):
        d = data_temp[i]
        seq_lens_train.append(len(d))
        for feature in features:
            X_train.append(torch.Tensor(d[feature]))

    # 真正的data
    seq_lens_train.append(len(data))
    for feature in features:
        X_train.append(torch.Tensor(data[feature].values))

    # pad
    X_train = pad_sequence(X_train, batch_first=True)

    # delete the temp_feature
    X_train = X_train[1:]

    # 将第0维度 分割开。 (samples * feature, length) --> (samples, feature, length)
    num_samples = len(data_temp)+1
    X_train = torch.chunk(X_train, chunks=num_samples, dim=0)  # chunk 后 成为tuple 了 # (tensor, tensor,...)

    # 格式转化 (tensor, tensor,...) --> [numpy, numpy,...] -->  tensor[]
    X_train_tensor = []
    for i in X_train:
        X_train_tensor.append(i.numpy())
    # todo 先array 后 Tensor 速度快。
    # X_train_tensor = np.array(X_train_tensor)
    X_train_tensor = torch.Tensor(X_train_tensor)

    # 交换维度 permute,
    X_train = X_train_tensor.permute(0, 2, 1)
    y_train = torch.tensor(y_train, dtype=torch.int64)

    seq_lens_train = torch.tensor(seq_lens_train, dtype=torch.int64)

    X_test, y_test, seq_lens_test = X_train, y_train, seq_lens_train
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test, seq_lens_test)

    # print("test_dataset", test_dataset)

    return test_dataset


# def testData_load(max_seq_len, data_folder_path):
#     data = clean(path_read=data_folder_path, max_seq_len=max_seq_len)
#     test_dataset = load_EV_datasets(data=data, max_seq_len=max_seq_len)
#     # print("test_dataset", test_dataset)
#     return test_dataset
def main(dataset, NUM_CLASSES, batch_size, MAX_SEQ_LEN, NUM_FEATURES, weights, data, car_Type):

    assert dataset in NUM_CLASSES.keys()

    max_seq_len = MAX_SEQ_LEN[dataset]
    test_dataset = load_EV_datasets(data, max_seq_len, car_Type)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    mlstm_fcn_model = MLSTMfcn(num_classes=NUM_CLASSES[dataset],
                               max_seq_len=MAX_SEQ_LEN[dataset],
                               num_features=NUM_FEATURES[dataset])
    mlstm_fcn_model.load_state_dict(torch.load('weights/' + weights))
    mlstm_fcn_model.to(device)

    criterion = nn.NLLLoss()
    #
    res = validation(mlstm_fcn_model, test_loader, criterion, device)
    res = res.numpy().tolist()
    # print("res", res)

    res_gb = []
    for r in res:
        if r == 1:
            res_gb.append("good")
        else:
            res_gb.append("bad")
    # print("predict label:", res_gb)

    return res_gb

    # test_loss, accuracy = validation(mlstm_fcn_model, test_loader, criterion, device)
    # print("Test loss: {:.6f}.. Test Accuracy: {:.2f}%".format(test_loss, accuracy*100))


def muhao_method(data_folder_path, car_Type):
    NUM_CLASSES = {'EV': 2, }
    NUM_FEATURES = {'EV': 5, }
    batch_size = 100
    dataset = 'EV'

    MAX_SEQ_LEN = None
    if car_Type == "A":
        weights = "TypeA_sample200_[5features].pt"
        MAX_SEQ_LEN = {'EV': 4385, }  # typeA
        folder = "./temp_data2/typeA/"
    if car_Type == "B":
        weights = "TypeB_sample200_[5features].pt"
        MAX_SEQ_LEN = {'EV': 4718, }  # typeB
        folder = "./temp_data2/typeB/"

    data = clean(data_folder_path, MAX_SEQ_LEN['EV'])
    # test_dataset = load_EV_datasets(data=data, max_seq_len=MAX_SEQ_LEN['EV'])

    flag = main(dataset, NUM_CLASSES, batch_size, MAX_SEQ_LEN, NUM_FEATURES, weights, data, car_Type)[-1]
    if flag == "bad":
        flag = 1
    else:
        flag = 0

    return flag


data_folder_path = r'./Test Data/DATA_1_10/LNBFFF3V4QA000001/'
# data_folder_path = r'./Test Data/DATA_11_20/LNBFFF3V4QA000020/raw_csv/'
# data_folder_path = r'./Test Data/DATA_21_30/LNBFFF3V4QA000022/raw_csv/'

car_Type = 'A'
# car_Type = 'B'

flag = muhao_method(data_folder_path, car_Type)
print("flag", flag)