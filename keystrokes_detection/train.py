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
        self.keystrokes = []
        self.up_times = []
        self.down_time = []

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
        if self.c_sharp:
            df_list = self.split_line(self.dataset_csv_file_address)
            for i in range(int((len(df_list)) / 2)):
                self.keystrokes.append(df_list[2 * i][2])
                self.up_times.append(df_list[2 * i + 1][0])
                self.down_time.append(df_list[2 * i][0])
        else:

            df = pd.read_csv(self.dataset_csv_file_address)
            df_list = df.values.tolist()

            for i in range(1, int((len(df_list)))):
                if df_list[i][2] != ' ':
                    self.keystrokes.append(df_list[i][3])
                else:
                    self.keystrokes.append(0)

                self.up_times.append(df_list[i][1])
                self.down_time.append(df_list[i][0])

        for i in range(len(self.keystrokes) - 1):
            # print(self.keystrokes[i], self.keystrokes[i + 1], self.down_time[i + 1] - self.up_times[i])
            if self.keystrokes[i] not in self.first_keys:
                print("NOT IN:", self.keystrokes[i])
                self.keystrokes[i] = 'Space'

            if self.keystrokes[i + 1] not in self.first_keys:
                # print("NOT IN2:", (self.keystrokes[i+1]))
                self.keystrokes[i + 1] = 'Space'
            # Note: change according to Dr. Rahaeimehr's idea:
            # self.data.append([self.keystrokes[i], self.keystrokes[i + 1], self.down_time[i + 1] - self.up_times[i]])
            # delta = ((self.down_time[i+ 1] + self.up_times[i + 1]) / 2) - ((self.down_time[i] + self.up_times[i]) / 2)
            delta = (self.down_time[i + 1] - self.down_time[i])
            self.data.append([self.keystrokes[i], self.keystrokes[i + 1], delta])
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
