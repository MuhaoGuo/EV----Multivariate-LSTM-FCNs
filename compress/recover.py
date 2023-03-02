import time
import zlib
import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
# -------------- int.txt 复原 -------------------:
def recover_int_txt(int_text, int_columns):
    columns_text = int_text.split("\n")[:-1]    # string of columns
    pre_restlts = int_text.split("\n")[-1]      # string of no_int_column + pred_res
    index1, index2 = pre_restlts.index("["), pre_restlts.index("]")
    no_int_columns_str, predict_res_str = pre_restlts[index1:index2+1], pre_restlts[index2+1:]
    no_int_columns = ast.literal_eval(no_int_columns_str)  # string of list -->  list
    new_int_columns = []

    if no_int_columns:
        for c in int_columns:
            if c not in no_int_columns:
                new_int_columns.append(c)
        int_columns = new_int_columns

    d = {}
    for i, column in enumerate(int_columns):
        column_text = columns_text[i]
        if column == "time":
            a_column = []
            for i, item in enumerate(column_text.split('o')):
                if not item:
                    continue
                start_time, distance, freq = item.split('d')[0], item.split('d')[1].split('f')[0], item.split('d')[1].split('f')[1]
                # print(start_time, distance, freq)
                if i == 0:
                    # a_column.extend([(int(start_time) + f * int(distance)) * 1000 for f in range(int(freq)+1)])
                    a_column.extend([(int(start_time) + f * int(distance)) for f in range(int(freq)+1)])
                else:
                    # a_column.extend([(int(start_time) + f * int(distance)) * 1000 for f in range(1, int(freq) + 1)])
                    a_column.extend([(int(start_time) + f * int(distance)) for f in range(1, int(freq) + 1)])

        else: # 包括正常 int 和 SOC, mile_GPS
            column_num_freq = []
            for item in column_text.split('o'):
                if not item:
                    continue
                if "f" in item:
                    column_num_freq.append(item.split("f"))
                else:
                    column_num_freq.append([item, '1'])
            a_column = []
            for [i, freq] in column_num_freq:
                a_column.extend([i for _ in range(int(freq))])
        d[column] = a_column

    output = pd.DataFrame.from_dict(d)
    return output

# -------------- float.txt 复原 -------------------:
def recover_float_txt(float_text, float_columns, output_dir_path=None):
    columns_text = float_text.split("\n")[:-1]
    no_float_columns_str = float_text.split("\n")[-1]
    no_float_columns = ast.literal_eval(no_float_columns_str)  # string of list -->  list

    new_float_columns = []
    if no_float_columns:
        for c in float_columns:
            if c not in no_float_columns:
                new_float_columns.append(c)
        float_columns = new_float_columns

    d = {}
    for i, column in enumerate(float_columns):
        column_text = columns_text[i]
        column_num_freq = []
        for item in column_text.split("o"):
            if not item:
                continue
            if "f" in item:
                column_num_freq.append(item.split("f"))
            else:
                column_num_freq.append([item, '1'])

        a_column = []
        for [i, freq] in column_num_freq:
            a_column.extend([i for _ in range(int(freq))])
        d[column] = a_column

    diff_diff = pd.DataFrame.from_dict(d)
    diff_diff_array = np.array(diff_diff).astype("int")

    # 每行的差值复原
    for row in diff_diff_array:
        for i in range(1, len(row)):
            row[i] += row[i-1]
    diff_array = diff_diff_array.T

    # 每列的差值复原
    for row in diff_array:
        for i in range(1, len(row)):
            row[i] += row[i-1]
    array = diff_array.T

    # 输出
    output = pd.DataFrame(array, columns=float_columns)
    for column in float_columns:
        if car_type == "A":
            if column in ['V_car_1', 'I', 'energy', 'Qcharge']:
                output[[column]] /= 100
            else:
                output[[column]] /= 10000
        elif car_type == "B":
            if column in ['V_Cell_max', 'V_Cell_min']:
                output[[column]] /= 10000
            else:
                output[[column]] /= 100
    return output


def recover_algorithm(vin, car_type, input_file_path, output_dir_path):
    print("vin:{}, car type:{}".format(vin, car_type))
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    if car_type == "A":
        int_columns = ["time", "mile_GPS", 'code_carstatus', 'flag_slowcharge', 'code_operation',  "SOC",
                       'R_positive', 'R_negative','T_max', 'T_min',
                       'num_cell', 'mode_batterycharge', 'flag_balance',
                        'T_cell_1', 'T_cell_2', 'T_cell_3', 'T_cell_4','T_cell_5', 'T_cell_6', 'T_cell_7', 'T_cell_8', 'T_cell_9',
                       'IDX_maxV', 'IDX_maxTEMP', 'IDX_minTEMP', 'IDX_minV', 'mile_ODO', 'state_onboardmachine']
        float_columns = ['V_car_1', 'I', 'V_Cell_max', 'V_Cell_min',
                     'energy', 'Qcharge',
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

    if "temperature_feature" in input_file_path:
        with open(input_file_path, 'rb') as f_int_bytes:
            int_bytes = f_int_bytes.read()
            int_text = zlib.decompress(int_bytes).decode()
        print("Recovering the int features ... ")
        output1 = recover_int_txt(int_text, int_columns)
        print("Saving the int output ... ")
        output1.to_csv(output_dir_path + "{}_{}_int.csv".format(vin, car_type))
        # print(output1)

    # read float.bytes
    if "voltage_feature" in input_file_path:
        with open(input_file_path, 'rb') as f_float_bytes:
            float_bytes = f_float_bytes.read()
            float_text = zlib.decompress(float_bytes).decode()
        print("Recovering the float features ... ")
        output2 = recover_float_txt(float_text, float_columns)
        print("Saving the float output ... ")
        output2.to_csv(output_dir_path + "{}_{}_float.csv".format(vin, car_type))



# input_dir_path = "./20221212_WY_feature/typeB/"
input_dir_path = "./output_newcar21/"

data_file_list = sorted(os.listdir(input_dir_path)) # 所有文件一起测试。
if data_file_list[0] == '.DS_Store':
    data_file_list = data_file_list[1:]
print(data_file_list)
# data_file_list = ['LB1F1T1EJG000016_B_temperature_feature.bytes', 'LB1F1T1EJG000016_B_voltage_feature.bytes', 'LB1F1T1EJG000021_B_temperature_feature.bytes', 'LB1F1T1EJG000021_B_voltage_feature.bytes']
# output_dir_path = "./20221212_WY_feature/Recover_typeB/" # 输出 恢复的 .csv 文件夹 的位置
output_dir_path = "./Recover_newcar21/"

if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

start_time = time.time()

for i in tqdm(range(len(data_file_list))):
    print("\n")
    file_name = data_file_list[i]
    if file_name == "finished_car_pointer.txt":
        continue
    input_file_path = input_dir_path + file_name
    vin, car_type = input_file_path.split("/")[-1].split("_")[0], input_file_path.split("/")[-1].split("_")[1]
    recover_algorithm(vin=vin, car_type=car_type, input_file_path=input_file_path, output_dir_path=output_dir_path)

end_time = time.time()
print("Finish! The Total Cost Time is {} s".format(round(end_time - start_time)))



# for file_name in data_file_list:
#     if file_name == "finished_car_pointer.txt":
#         continue
#     input_file_path = input_dir_path + file_name
#     vin, car_type = input_file_path.split("/")[-1].split("_")[0], input_file_path.split("/")[-1].split("_")[1]
#     recover_algorithm(vin=vin, car_type=car_type, input_file_path=input_file_path, output_dir_path=output_dir_path)
#     print("-----------------")

# Finish! The Total Cost Time is 3082 s



