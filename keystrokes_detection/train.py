import os

import numpy as np
import xlsxwriter
import pandas as pd


class MatrixMaker:
    def __init__(self, csv_file_address='.', c_sharp=False, add_pre_data=True, output_address='./'):
        self.previous_data = None
        self.add_data = add_pre_data
        self.worksheet = None
        self.c_sharp = c_sharp
        self.dataset_csv_file_address = csv_file_address
        self.file_name = output_address
        self.data = []
        if self.add_data and os.path.isfile(f'{self.file_name}/model.xlsx'):
            self.previous_data = pd.read_excel(f'{self.file_name}/model.xlsx', sheet_name=f'data', header=0).to_numpy()
            for pre_data in self.previous_data:
                self.data.append(pre_data)

        # print(self.data)

        self.first_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                           'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                           'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                           'z', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                           ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
                           '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'Space']

        self.workbook = xlsxwriter.Workbook(f'{self.file_name}/model.xlsx')
        # print(f'{self.file_name}model.xlsx')

        self.worksheets = []
        #self.keystrokes = []
        #self.up_times = []
        #self.down_time = []

    def close_file(self):
        # print("file closed")
        self.workbook.close()

    def write_matrix_to_file(self):

        self.worksheet = self.workbook.add_worksheet(f"data")
        self.worksheet.write(0, 0, 'First_Key')
        self.worksheet.write(0, 1, 'Second_Key')
        self.worksheet.write(0, 2, 'Delta_Time')

        for index, row in enumerate(self.data):
            self.worksheet.write(index + 1, 0, row[0])
            self.worksheet.write(index + 1, 1, row[1])
            self.worksheet.write(index + 1, 2, row[2])

    def write_analysis_to_file(self):

        self.worksheet = self.workbook.add_worksheet(f"analysis")
        self.worksheet.write(0, 0, 'First_Key')
        self.worksheet.write(0, 1, 'Second_Key')
        self.worksheet.write(0, 2, 'Mean')
        self.worksheet.write(0, 3, 'STD')

        keys = [[subarray[0], subarray[1]] for subarray in self.data]
        # print(keys)
        registered_keys = []
        all_means = []
        all_stds = []
        data_index = 0
        for first_key, second_key in keys:
            if [first_key, second_key] not in registered_keys:
                registered_keys.append([first_key, second_key])
                elements = [subarray for index, subarray in enumerate(self.data) if
                            subarray[0] == first_key and subarray[1] == second_key]
                time_elements = [element[2] for element in elements]
                time_mean = np.mean(time_elements)
                time_std = np.std(time_elements)
                all_means.append(time_mean)
                all_stds.append(time_std)
                # print(f"({first_key} , {second_key}) - mean:{time_mean} - std:{time_std}")
                self.worksheet.write(data_index + 1, 0, first_key)
                self.worksheet.write(data_index + 1, 1, second_key)
                self.worksheet.write(data_index + 1, 2, time_mean)
                self.worksheet.write(data_index + 1, 3, time_std)

                data_index += 1

        means = np.mean(all_means)
        std_means = np.std(all_means)
        std_stds = np.std(all_stds)
        print(f"** Means: {means}, Std of means: {std_means}, std of stds:{std_stds} **")

    def train(self):
        df_list = self.split_line(self.dataset_csv_file_address)

        temp = []
        res = []

        for array in df_list:
            if array[1] == '0':
                temp.append(array)
            else:
                # Find the last array in temp with the same third field
                flag = False
                for i in reversed(range(len(temp))):
                    if temp[i][2] == array[2]:
                        res.append([temp[i][0], array[0], temp[i][2]])
                        del temp[i]
                        flag = True
                        break      
                if not flag:
                    res.append([0, array[0], array[2]])     
        if len(temp) > 0:
            for i in range(len(temp)):
                res.append([temp[i][0], math.inf, temp[i][2]]) 
        #sorted(res, key=lambda x: x[0])
        sorted(res)
        
        #for i in range(int((len(df_list)) / 2)):
        #    self.keystrokes.append(df_list[2 * i][2])
        #    self.up_times.append(df_list[2 * i + 1][0])
        #    self.down_time.append(df_list[2 * i][0])

        for i in range(0, len(res) - 1):

            if res[i][2] not in self.first_keys:
                print("NOT IN:", res[i][2])
                res[i][2] = 'Space'

            if res[i+1][2] not in self.first_keys:
                print("NOT IN:", res[i+1][2])
                res[i+1][2] = 'Space'

            delta = (res[i + 1][0] - res[i][0])
            self.data.append([res[i][2], res[i + 1][2], delta])
        self.write_matrix_to_file()
        self.write_analysis_to_file()
        self.close_file()

    def split_line(self, filename, split_char=','):
        # print(filename)
        lst = list()
        with open(filename) as file:
            for line in file:
                line = line.strip()
                elements = line.split(split_char)
                # print(elements)
                elements[0] = int(elements[0]) / 10000000     # Time
                elements[2] = chr(int(elements[2], 16))       # ascii (hex)
                elements[3] = int(elements[3], 16)            # scan code
                # print("elements:", elements)
                lst.append(elements)

        return lst


# if __name__ == '__main__':
#     """
#         help: please define "dataset_csv_file_address" and "maximum_number_of_repetitions"
#                 and run this python file to build your model
#     """
#
#     # dataset_csv_file_address = r"E:\alireza\augusta\codes\recorder\dataset\hani/random.txt"
#     dataset_csv_file_address = r"E:\alireza\augusta\codes\recorder\dataset\hani/text.txt"
#
#     MatrixMaker(dataset_csv_file_address, c_sharp= True, add_pre_data=True).train()
