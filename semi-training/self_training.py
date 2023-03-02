import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import MLSTMfcn
import time
from torch.nn.utils.rnn import pad_sequence
import argparse

NUM_CLASSES = 2
NUM_FEATURES = 5
MAX_SEQ_LEN = 1000

def load_EV_train_val_datasets(folder_input, car_type, val_size):
    file_list = sorted(os.listdir(folder_input)) # 要sort， 和 y_train 对应。
    if file_list[0] == ".DS_Store":
        file_list = file_list[1:]
    num_samples = len(file_list)
    features = ["V_Cell_max", "V_Cell_min", "T_max", "T_min", "SOC"]
    y_train_B = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] # 坏车 0， 好车1
    y_train_A = [0, 0, 1, 1, 1, 1]  # 0, 1, 1, 1 # typeA: ['01', '02', '03', '04', '05', '06', '11', '12', '13', '14']
    if car_type == 'A':
        y_train = y_train_A
    elif car_type =='B':
        y_train = y_train_B
    else:
        raise ("Wrong Type!")
    print("#samples:", num_samples)
    print("#features:", len(features))
    print("samples file_list", file_list)

    seq_lens_train = []  # 记录每个sample的 有效值，len的长度
    X_train = []
    for file in file_list:
        data = pd.read_csv(folder_input + file)
        seq_lens_train.append(len(data))
        sample = []
        for feature in features:
            sample.append(data[feature])
        X_train.append(sample)
    X_train = np.array(X_train)
    X_train = torch.Tensor(X_train)

    X_train = X_train.permute(0, 2, 1) # (#samples, #features, len) --> (#samples, len, #features)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    seq_lens_train = torch.tensor(seq_lens_train, dtype=torch.int64)

    if val_size != 0:
        # 分开 index_train, index_val
        index_all = [i for i in range(len(y_train))]   # 一共20个 type B 车
        index_train, index_val = train_test_split(index_all, test_size=val_size, random_state=42)
        print("(index_train,index_val)", (index_train, index_val))

        # 分开 X_train, X_val
        X_train, X_val = X_train[index_train], X_train[index_val]
        print("(X_train, X_val)", (X_train.shape, X_val.shape))

        # 分开 y_train, y_val
        y_train, y_val = y_train[index_train], y_train[index_val]
        print("(y_train, y_test)", (y_train, y_val))

        # 分开 seq_lens_train,  seq_lens_val
        seq_lens_train, seq_lens_val = seq_lens_train[index_train], seq_lens_train[index_val]
        print("(seq_lens_train, seq_lens_val)", (seq_lens_train, seq_lens_val))

        # 得到 train 和 val
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train, seq_lens_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val, seq_lens_val)
        return train_dataset, val_dataset

    elif val_size == 0:
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train, seq_lens_train)
        val_dataset = None
        return train_dataset, val_dataset

def validation(model, valloader, criterion, device):
    accuracy = 0
    val_loss = 0
    for inputs, labels, seq_lens in valloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs, seq_lens)
        val_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # print("labels.data", labels.data)
        # print("ps.max(1)[1]", ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    return val_loss, accuracy

def train(model, trainloader, validloader, criterion, optimizer, epochs, print_every, device, run_name):
    '''
    先 train labeled data
    后 train combined data
    '''
    valid_loss_min = np.Inf  # track change in validation loss
    steps = 0
    if not validloader:
        print("No val set")
        validloader = trainloader

    for e in range(epochs):
        train_loss = 0.0
        model.train()
        for inputs, labels, seq_lens in trainloader:
            steps += 1
            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs, seq_lens)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.6f}.. ".format(train_loss / print_every),
                      "Val Loss: {:.6f}.. ".format(valid_loss / len(validloader)),
                      "Val Accuracy: {:.2f}%".format(accuracy / len(validloader) * 100))

                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                    torch.save(model.state_dict(), run_name + '.pt')
                    valid_loss_min = valid_loss
                train_loss = 0
                model.train()

def self_training():
    epochs = 80
    batch_size = 10
    learning_rate = 0.01
    val_size = 0.0
    # 第一次 训练 A 车 ---------------------
    folder_input = "./Preprocess_data_semi/typeA_labeled/"  # 有 label 的 data
    car_type = 'A'
    save_folder = "./weights/"
    save_filename = "TypeA_[5features][no_val][len=200][bs=10][Semi_labeled]"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    # 加载labeled dataset
    train_dataset, val_dataset = load_EV_train_val_datasets(folder_input=folder_input, car_type=car_type, val_size=val_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    for batch in train_loader:
        print("batch: X.shape:{} | y.shape:{} | length.shape:{}".format(batch[0].shape, batch[1].shape, batch[2].shape))
        break

    # 初始化模型
    mlstm_fcn_model = MLSTMfcn(num_classes=NUM_CLASSES, max_seq_len=MAX_SEQ_LEN, num_features=NUM_FEATURES)
    mlstm_fcn_model.to(device)
    # optimizer = optim.SGD(mlstm_fcn_model.parameters(), lr=args.learning_rate, momentum=0.9)
    optimizer = optim.Adam(mlstm_fcn_model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # 训练 model based on labeled data, 训练好后，自动加载到weights中。
    print("Training ... ------------")
    train(mlstm_fcn_model,
          train_loader,
          val_loader,
          criterion,
          optimizer,
          epochs=epochs,
          print_every=1,
          device=device,
          run_name=save_folder + save_filename)

    # 加载 unlabeled data ---------------------
    unlabeled_data_dir = "./Preprocess_data_semi/typeA_unlabeled/"
    file_list = sorted(os.listdir(unlabeled_data_dir)) # 要sort， 和 y_train 对应。
    features = ["V_Cell_max", "V_Cell_min", "T_max", "T_min", "SOC"]

    seq_lens_train = []  # 记录每个sample的 有效值，len的长度
    X_train = []
    for file in file_list:
        if file == ".DS_Store":
            continue
        data = pd.read_csv(unlabeled_data_dir + file)
        seq_lens_train.append(len(data))
        sample = []
        for feature in features:
            sample.append(data[feature])
        X_train.append(sample)
    X_train = np.array(X_train)
    X_train = torch.Tensor(X_train)
    X_train = X_train.permute(0, 2, 1) # 交换维度 permute: (#samples, #features, len) --> (#samples, len, #features)
    seq_lens_train = torch.tensor(seq_lens_train, dtype=torch.int64)
    train_dataset_unlabeled = torch.utils.data.TensorDataset(X_train, seq_lens_train)  # 构建unlabeled dataset
    train_loader_unlabeled = torch.utils.data.DataLoader(train_dataset_unlabeled, batch_size=len(train_dataset_unlabeled))

    # 加载 模型  ---------------------
    car_Type = "A"
    if car_Type == "A":
        weight_path = "./weights/TypeA_[5features][no_val][len=200][bs=10][Semi_labeled].pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlstm_fcn_model = MLSTMfcn(num_classes=NUM_CLASSES,
                               max_seq_len=MAX_SEQ_LEN,
                               num_features=NUM_FEATURES)
    mlstm_fcn_model.load_state_dict(torch.load(weight_path))
    mlstm_fcn_model.to(device)

    # unlabeled data 预测
    for X, L in train_loader_unlabeled:
        continue
    output = mlstm_fcn_model.forward(X, L)
    ps = torch.exp(output)
    sudo_labels = ps.max(1)[1].numpy()

    # 合并 labeled & unlabeled dataset ---------------------
    new_X, new_Y, new_L = [], [], []
    for sample in train_dataset:
        X, y, l = sample[0].numpy(), sample[1].numpy(), sample[2].numpy()
        new_X.append(X)
        new_Y.append(y)
        new_L.append(l)
    for sample_unlabeled in train_dataset_unlabeled:
        X, l = sample_unlabeled[0].numpy(), sample_unlabeled[1].numpy()
        new_X.append(X)
        new_L.append(l)
    new_Y.extend(sudo_labels)

    new_X = np.array(new_X)
    new_Y = np.array(new_Y)
    new_L = np.array(new_L)
    new_X = torch.tensor(new_X)
    new_Y = torch.tensor(new_Y)
    new_L = torch.tensor(new_L)

    train_dataset_combined = torch.utils.data.TensorDataset(new_X, new_Y, new_L)
    train_loader_combined = torch.utils.data.DataLoader(train_dataset_combined, batch_size=10)

    # 重新初始化一个model 用于重新训练。
    mlstm_fcn_model_new = MLSTMfcn(num_classes=NUM_CLASSES, max_seq_len=MAX_SEQ_LEN, num_features=NUM_FEATURES)
    mlstm_fcn_model_new.to(device)
    optimizer = optim.Adam(mlstm_fcn_model_new.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # 训练 combined data
    save_filename = "TypeA_[5features][no_val][len=200][bs=10][Semi_combined].pt"
    print("Training combined data ... ------------")
    train(model=mlstm_fcn_model_new,
          trainloader = train_loader_combined,
          validloader = None,
          criterion = criterion,
          optimizer = optimizer,
          epochs=epochs,
          print_every=1,
          device=device,
          run_name=save_folder + save_filename)




self_training()

