import os
from utils import clean

MAX_SEQ_LEN = 1000
ANALYSIS_DAY = 1000

def prepocess_data(folder_input, folder_output):
    '''
    做 data preprocess。产生的 data: 用于 training 过程。
    folder_input, folder_output 都是文件夹。

    typeB = ['07','08', '09', '10', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    labels_B = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # type B 坏车 0：[ 19, 20, 21]
    typeA =  ['01', '02', '03', '04', '05', '06', '11', '12', '13', '14']
    labels_A =  [0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    # A 坏车 0 : ["01", "02", "11"]
    '''
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    car_list = sorted(os.listdir(folder_input))
    # print("car_list:", car_list)

    for car_vin in car_list:
        if car_vin == '.DS_Store':
            continue
        a_car_path = folder_input + car_vin + "/raw_csv/"
        car_clean = clean(a_car_path, max_seq_len=MAX_SEQ_LEN, analysis_day=ANALYSIS_DAY)
        print("{} --> cleaned len: {}".format(car_vin, len(car_clean)))

        # data_one_month_without_outliers = data_one_month[(data_one_month['T_max'] <= 100) & (data_one_month['T_min'] <= 100) & (data_one_month['V_Cell_max'] <= 5.0) & (data_one_month['V_Cell_min'] <= 5.0)]
        car_clean.to_csv(folder_output + car_vin + ".csv")  # 存入新文件(folder_output)中。


prepocess_data(
    folder_input="./raw_DATA_A_B/typeA/",
    folder_output="./Preprocess_data/typeA/",
    )
prepocess_data(
    folder_input="./raw_DATA_A_B/typeB/",
    folder_output="./Preprocess_data/typeB/",
    )




