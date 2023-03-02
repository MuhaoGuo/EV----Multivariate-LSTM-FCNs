import os.path
from statistic_send import algorithm
from tqdm import tqdm
import sys
# import logging

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def main(input_dir_path, output_dir_path):

    sys.stdout = Logger("stdout.log", sys.stdout)
    sys.stderr = Logger("stdin.log", sys.stderr)  # redirect std err, if necessary

    car_data_file_list = sorted(os.listdir(input_dir_path))
    if car_data_file_list[0] == '.DS_Store':
        car_data_file_list = car_data_file_list[1:]
    print("car_data_file_list:", car_data_file_list)
    # print("COUNT", len(car_data_file_list))

    if os.path.exists(output_dir_path + "finished_car_pointer.txt"):
        with open(output_dir_path + "finished_car_pointer.txt", "r") as f:
            finished_car_pointer = f.read().split()
        ptr = int(finished_car_pointer[0])
    else:
        ptr = -1

    for i in tqdm(range(len(car_data_file_list[:]))):
        if i <= ptr:
            continue
        car_vin = car_data_file_list[i]   # each vin like: LNBFFF3V4QA000001
        try:
            input_dir_path_1 = input_dir_path + car_vin + "/raw_csv/"
            algorithm(input_dir_path=input_dir_path_1, output_dir_path=output_dir_path)
            with open(output_dir_path + "finished_car_pointer.txt", "w") as f:
                f.write(str(i) + "\n" + str(car_vin))

        except Exception as e:
            print("\n", e)

main(input_dir_path = "./raw_DATA/" ,
     output_dir_path ="./raw_DATA_OUTPUTS/"
)



# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("debug.log"),
#         logging.StreamHandler()
#     ]
# )




# floatdata = pd.read_csv("./data_newcar21/LB1F1T1EJG000021_B_float.csv")
# intdata = pd.read_csv("./data_newcar21/LB1F1T1EJG000021_B_int.csv")
#
# newcar_21 = pd.concat([floatdata, intdata], axis=1,)
# newcar_21.to_csv("./data_newcar21/LB1F1T1EJG000021.csv")
# print(newcar_21)
# print(len(newcar_21))


