import os
import numpy as np
import pandas as pd
import zlib
import pickle
from sklearn import preprocessing
import time

# load our model
with open('model.pkl', 'rb') as f:
    our_model = pickle.load(f)

def help(array, is_int):
    res = ""
    pre_num = array[0]
    counts = 0
    for cur_num in array:
        if cur_num == pre_num:
            counts += 1
        else:
            if is_int:
                if counts == 1:
                    res += str(round(pre_num)) + "o"
                else:
                    res += str(round(pre_num)) + "f" + str(counts) + "o"
            else:
                if counts == 1:
                    res += str(pre_num) + "o"
                else:
                    res += str(pre_num) + "f" + str(counts) + "o"
            pre_num = cur_num
            counts = 1

    if is_int:
        if counts == 1:
            res += str(round(pre_num)) + "o"
        else:
            res += str(round(pre_num)) + "f" + str(counts)+ "o"
    else:
        if counts == 1:
            res += str(pre_num) + "o"
        else:
            res += str(pre_num) + "f" + str(counts) + "o"
    return res

def help_time(array, is_int=True):
    res = ""
    counts = 0
    # pre_distance = array[1]//1000 - array[0]//1000
    pre_distance = array[1] - array[0]
    # start_val = array[0]//1000
    start_val = array[0]
    for i in range(1, len(array)):
        # cur_distance = array[i]//1000 - array[i-1]//1000
        cur_distance = array[i] - array[i-1]
        if cur_distance == pre_distance:
            counts += 1
        else:
            res += str(start_val) + "d" + str(pre_distance) + "f" + str(counts) + "o"
            # start_val = array[i-1]//1000
            start_val = array[i-1]
            pre_distance = cur_distance
            counts = 1
    res += str(start_val) + "d" + str(pre_distance) + "f" + str(counts) + "o"
    return res

def int_features_processing(car_dataframes_int, int_columns):
    out_put = ""
    for column in int_columns:
        is_int = False if column == "SOC" or column == "mile_GPS" else True
        column_i = car_dataframes_int[[column]]
        column_i = np.array(column_i).ravel()
        if column == "time":
            output = help_time(column_i, is_int=is_int)
        else:
            output = help(column_i, is_int=is_int)
        out_put += output + "\n"
    return out_put

def float_feature_processing(car_dataframes_float, float_columns, car_type):
    dataframes_float_new = pd.DataFrame() # 要重建一个，避免 改变 输入变量 car_dataframes_float
    # 先乘法 float -- > int：
    for column in float_columns:
        if car_type == "A":
            if column in ['V_car_1', 'I', 'energy', 'Qcharge']:
                dataframes_float_new[column] = car_dataframes_float[column].apply(lambda x: round(x*100))
            else:
                dataframes_float_new[column] = car_dataframes_float[column].apply(lambda x: round(x*10000))

        elif car_type == "B":
            if column in ['V_Cell_max', 'V_Cell_min']:
                dataframes_float_new[column] = car_dataframes_float[column].apply(lambda x: round(x*10000))
            else:
                dataframes_float_new[column] = car_dataframes_float[column].apply(lambda x: round(x*100))

    dataframes_float_new = dataframes_float_new.astype(int)
    dataframes_float_new.reset_index(drop= True, inplace=True)

    # 按列方向 取差值
    first_0 = pd.DataFrame([[0 for _ in range(len(float_columns))]], columns=float_columns)  # 构造一个全 0 row，index =0
    df1 = pd.concat([first_0, dataframes_float_new], ignore_index=True).iloc[:-1, :]  # 省略最后一个row
    diff = dataframes_float_new - df1

    # 按行方向 取差值
    diff.insert(0, "temp", [0 for _ in range(len(diff))])    # 增加1列 在第0列
    diff_diff = diff.iloc[:, 1:] - diff.iloc[:, :-1].values  # 按行方向 取差值

    DF = diff_diff
    out_put = ""
    for column in float_columns:
        is_int = True
        column_i = DF[[column]]
        column_i = np.array(column_i).ravel()
        output = help(column_i, is_int=is_int)
        out_put += output + "\n"
    return out_put

def algorithm(input_dir_path, output_dir_path):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    time1 = time.time()
    # 几个data文件合起来
    data_file_list = sorted(os.listdir(input_dir_path))
    if data_file_list[0] == '.DS_Store':
        data_file_list = data_file_list[1:]
    car_data_list = []
    print("data_file_list", data_file_list)
    for i in range(len(data_file_list)):
        car_data_list.append(pd.read_csv(input_dir_path + data_file_list[i]))
    car_dataframes = pd.concat(car_data_list, ignore_index=True)
    # print("car_dataframes", car_dataframes)

    # 判断 car type 和 vin
    car_type = "B" if "V_cell_91" in car_dataframes.columns else "A"
    vin = car_dataframes["vin"][0]
    # print("vin", vin)

    time2 = time.time()

    # drop 没用的值
    car_dataframes = car_dataframes.dropna(axis=0, how="any")
    # car_dataframes = car_dataframes.loc[(car_dataframes["V_Cell_max"] != 0) | (car_dataframes["V_Cell_min"] != 0)]
    car_dataframes = car_dataframes.loc[~(car_dataframes.iloc[:, 4:-5] == 0).all(axis=1)] # drop zeros roughly
    car_dataframes = car_dataframes.drop(['vin'], axis=1)
    print("\nvin: {} | car type: {} | length: {}".format(vin, car_type, len(car_dataframes)))

    # 分为 int 和 float
    if car_type == "A":
        int_columns = ["time", "mile_GPS", 'code_carstatus', 'flag_slowcharge', 'code_operation',  "SOC",
                       'R_positive', 'R_negative','T_max', 'T_min',
                       'num_cell', 'mode_batterycharge', 'flag_balance',
                        'T_cell_1', 'T_cell_2', 'T_cell_3', 'T_cell_4','T_cell_5', 'T_cell_6', 'T_cell_7', 'T_cell_8', 'T_cell_9',
                       'IDX_maxV', 'IDX_maxTEMP', 'IDX_minTEMP', 'IDX_minV', 'mile_ODO', 'state_onboardmachine']
        float_columns = ['V_car_1', 'I', 'V_Cell_max', 'V_Cell_min','energy', 'Qcharge',
                     'V_cell_1','V_cell_2', 'V_cell_3', 'V_cell_4', 'V_cell_5', 'V_cell_6', 'V_cell_7',
                     'V_cell_8', 'V_cell_9', 'V_cell_10', 'V_cell_11', 'V_cell_12', 'V_cell_13', 'V_cell_14',
                     'V_cell_15', 'V_cell_16', 'V_cell_17', 'V_cell_18', 'V_cell_19', 'V_cell_20', 'V_cell_21',
                     'V_cell_22', 'V_cell_23', 'V_cell_24', 'V_cell_25', 'V_cell_26', 'V_cell_27', 'V_cell_28',
                     'V_cell_29', 'V_cell_30', 'V_cell_31']
    if car_type == "B":
        int_columns = ['time', 'mile_GPS', 'code_carstatus', 'flag_fastcharge', 'flag_slowcharge', 'code_operation',
                       'state_charge', 'SOC',  'R_positive', 'R_negative', 'T_max', 'T_min', 'flag_balance',
                       'mode_batterycharge', 'num_cell',  'num_tempsensors',
                       'T_cell_1', 'T_cell_2', 'T_cell_3', 'T_cell_4', 'T_cell_5', 'T_cell_6', 'T_cell_7', 'T_cell_8',
                       'T_cell_9', 'T_cell_10', 'T_cell_11', 'T_cell_12', 'T_cell_13', 'T_cell_14', 'T_cell_15',
                       'T_cell_16', 'T_cell_17', 'T_cell_18', 'T_cell_19', 'T_cell_20', 'T_cell_21', 'T_cell_22',
                       'T_cell_23', 'T_cell_24',
                       'IDX_maxV', 'IDX_maxTEMP', 'IDX_minTEMP', 'IDX_minV', 'state_onboardmachine', 'mile_ODO'
                       ]
        float_columns = ['V_car_1', 'I', 'V_Cell_max', 'V_Cell_min',  'energy', 'Qcharge',
                         'V_cell_1', 'V_cell_2', 'V_cell_3', 'V_cell_4', 'V_cell_5', 'V_cell_6', 'V_cell_7', 'V_cell_8',
                         'V_cell_9', 'V_cell_10', 'V_cell_11', 'V_cell_12', 'V_cell_13', 'V_cell_14', 'V_cell_15',
                         'V_cell_16', 'V_cell_17', 'V_cell_18', 'V_cell_19', 'V_cell_20', 'V_cell_21', 'V_cell_22',
                         'V_cell_23', 'V_cell_24', 'V_cell_25', 'V_cell_26', 'V_cell_27', 'V_cell_28', 'V_cell_29',
                         'V_cell_30', 'V_cell_31', 'V_cell_32', 'V_cell_33', 'V_cell_34', 'V_cell_35', 'V_cell_36',
                         'V_cell_37', 'V_cell_38', 'V_cell_39', 'V_cell_40', 'V_cell_41', 'V_cell_42', 'V_cell_43',
                         'V_cell_44', 'V_cell_45', 'V_cell_46', 'V_cell_47', 'V_cell_48', 'V_cell_49', 'V_cell_50',
                         'V_cell_51', 'V_cell_52', 'V_cell_53', 'V_cell_54', 'V_cell_55', 'V_cell_56', 'V_cell_57',
                         'V_cell_58', 'V_cell_59', 'V_cell_60', 'V_cell_61', 'V_cell_62', 'V_cell_63', 'V_cell_64',
                         'V_cell_65', 'V_cell_66', 'V_cell_67', 'V_cell_68', 'V_cell_69', 'V_cell_70', 'V_cell_71',
                         'V_cell_72', 'V_cell_73', 'V_cell_74', 'V_cell_75', 'V_cell_76', 'V_cell_77', 'V_cell_78',
                         'V_cell_79', 'V_cell_80', 'V_cell_81', 'V_cell_82', 'V_cell_83', 'V_cell_84', 'V_cell_85',
                         'V_cell_86', 'V_cell_87', 'V_cell_88', 'V_cell_89', 'V_cell_90', 'V_cell_91']

    time3 = time.time()
    # 判断有没有具体的 columns:
    no_int_columns = []
    no_float_columns = []
    new_int_columns = []
    new_float_columns = []
    for int_column in int_columns:
        if int_column in list(car_dataframes.columns):
            new_int_columns.append(int_column)
        else:
            no_int_columns.append(int_column)
    for float_column in float_columns:
        if float_column in list(car_dataframes.columns):
            new_float_columns.append(float_column)
        else:
            no_float_columns.append(float_column)
    int_columns = new_int_columns
    float_columns = new_float_columns

    car_dataframes_int = car_dataframes[int_columns]
    car_dataframes_float = car_dataframes[float_columns]

    time4 = time.time()
    ### int 和 float 分别处理
    print("Int features processing...")
    mytext_int = int_features_processing(car_dataframes_int, int_columns)
    time5 = time.time()
    print("Float features processing...")
    mytext_float = float_feature_processing(car_dataframes_float, float_columns, car_type)
    time6 = time.time()

    # 特征提取：
    T_max = mytext_int.split("\n")[8] if car_type == "A" else mytext_int.split("\n")[10]
    selected_features = T_max.split("o")[:11]
    selected_features = [feature.split("f") if "f" in feature else feature.split("f")+["1"] for feature in selected_features]
    selected_features = np.array(selected_features).astype(int).tolist()
    min_max_scaler = preprocessing.MinMaxScaler()
    test_features = min_max_scaler.fit_transform(selected_features)

    # 模型预测：
    res_list = our_model.predict(test_features).tolist()
    result = 1 if res_list.count(1) >= res_list.count(0) else 0
    print("Predicting Result:{} -> {}".format(result, "Bad" if result == 0 else "Good"))
    mytext_int += str(no_int_columns) + "predict result: {}".format(result)
    mytext_float += str(no_float_columns)

    time7 = time.time()
    # 输出 output
    bytes_int = zlib.compress(mytext_int.encode())
    with open(output_dir_path + '{}_{}_temperature_feature.bytes'.format(vin, car_type), 'wb') as fbinary:
        fbinary.write(bytes_int)
    time8 = time.time()

    bytes_float = zlib.compress(mytext_float.encode())
    with open(output_dir_path + '{}_{}_voltage_feature.bytes'.format(vin, car_type), 'wb') as fbinary:
        fbinary.write(bytes_float)
    time9 = time.time()

    print("combine files: {}".format(time2 - time1))
    print("drop nan: {}".format(time3 - time2))
    print("judge value columns: {}".format(time4 - time3))
    print("int_features processing: {}".format(time5 - time4))
    print("float features processing: {}".format(time6 - time5))
    print("model predict: {}".format(time7 - time6))
    print("output T feature: {}".format(time8 - time7))
    print("output V feature: {}".format(time9 - time8))
