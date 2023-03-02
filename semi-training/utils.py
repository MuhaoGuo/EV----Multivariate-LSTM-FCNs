import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import datetime



def clean(path_read, max_seq_len, analysis_day):
    '''
    针对 training data 和 之后的test data 都是一样的处理方式。
    针对每一辆车：合并CSV --> 选择时间 --> Drop 0 and NAN --> 选取 features --> MAX_POINTS 多退少补 --> sampling
    path_read: 一辆车的path
    max_seq_len: 提前定义的。
    analysis_day: 多少天
    '''
    MAX_POINTS = 200000

    car_data_file_list = sorted(os.listdir(path_read))
    if car_data_file_list[0] == '.DS_Store':
        car_data_file_list = car_data_file_list[1:]

    # 7个data文件合起来
    car_dataframes = []
    for i in range(len(car_data_file_list)):
        car_dataframes.append(pd.read_csv(os.path.join(path_read, car_data_file_list[i])))
    car = pd.concat(car_dataframes)

    # 选择时间
    car['time'] = pd.to_datetime(car['time']) # time to real-time ???
    car.sort_values(by='time', inplace=True)
    car = car.set_index('time')
    window = (car.index[-1] + datetime.timedelta(days=-analysis_day)).date()
    car = car.loc[window:, :]
    car = car.reset_index(drop=False)
    # print(car)

    # drop 0 and NAN
    car = car.loc[~(car.iloc[:, 4:-5] == 0).all(axis=1)] # drop zeros roughly
    car = car.dropna(axis=0, how="any")     # clean data drop nan

    # 选取 features
    # useful_features = ['time', 'V_Cell_max', 'V_Cell_min', 'T_max', 'T_min', 'SOC']
    useful_features = ['V_Cell_max', 'V_Cell_min', 'T_max', 'T_min', 'SOC']
    car = pd.DataFrame(car, columns=useful_features)

    # 使得每个 car 都是 MAX_POINTS 相同的长度。
    if len(car) >= MAX_POINTS:
        car = car[:MAX_POINTS]
    else:
        zeros = np.zeros((MAX_POINTS-len(car), len(useful_features)))
        zeros_dataframe = pd.DataFrame(zeros, columns=useful_features)
        car = pd.concat([zeros_dataframe, car], axis=0)

    # sampling
    while len(car) > max_seq_len:
        car = car[::10]

    car.index = range(len(car.index))
    return car

























