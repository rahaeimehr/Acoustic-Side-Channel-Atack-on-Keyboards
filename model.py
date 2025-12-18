import os
import math
import numpy as np
import xlsxwriter
import pandas as pd
import consts
from my_utilities import TitlePrint as tp, ColorPrint as cp, rep_code
from pathlib import Path

class Model:
    def __init__(self, meta_data_file, model_file = None,  append_to_model = True):
        self.previous_data = None
        self.worksheet = None
        self.meta_data_file = Path(meta_data_file)

        if model_file is None:
            self.model_file = self.meta_data_file.parent / "model.xlsx"
        else:
            self.model_file = Path(model_file)
            

        self.workbook = xlsxwriter.Workbook(self.model_file)
        self.worksheets = []

        self.data = []


        if append_to_model and os.path.isfile(self.model_file):
            self.previous_data = pd.read_excel(self.model_file, sheet_name=f'data', header=0).to_numpy()
            for pre_data in self.previous_data:
                self.data.append(pre_data)

    def close_file(self):
        self.workbook.close()

    def write_model_to_file(self):

        self.worksheet = self.workbook.add_worksheet(f"data")
        self.worksheet.write(0, 0, 'First_Key')
        self.worksheet.write(0, 1, 'Second_Key')
        self.worksheet.write(0, 2, 'Delta_Time')
        self.worksheet.write(0, 3, 'First_Key_Rep')
        self.worksheet.write(0, 4, 'Second_Key_Rep')

        for index, row in enumerate(self.data):
            self.worksheet.write(index + 1, 0, row[0])
            self.worksheet.write(index + 1, 1, row[1])
            self.worksheet.write(index + 1, 2, row[2])
            self.worksheet.write(index + 1, 3, row[3])
            self.worksheet.write(index + 1, 4, row[4])

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
        df_list = self.split_line(self.meta_data_file)

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
                        if len(temp) > 1: cp.Warning(temp)
                        del temp[i]                        
                        flag = True
                        break      
                if not flag:
                    res.append([0, array[0], array[2]])     

        for i in range(len(temp)):
            res.append([temp[i][0], math.inf, temp[i][2]]) 

        #sorted(res, key=lambda x: x[0])

        sorted(res)
        
        for i in range(len(res)):            
            ch = chr(res[i][2])
            if ( ch not in consts.KEYS) and (ch not in consts.DELIMITERS):
                cp.Error( f'NOT IN keys: {i} , code: {res[i][2]}, char : {repr(ch)}')

        for i in range(0, len(res) - 1):
            delta = (res[i + 1][0] - res[i][0])
            self.data.append([res[i][2], res[i + 1][2], delta , rep_code(res[i][2]), rep_code(res[i + 1][2])] )

        self.write_model_to_file()
        # self.write_analysis_to_file()
        self.close_file()

    def split_line(self, filename, split_char=','):
        # print(filename)
        lst = list()
        with open(filename) as file:
            for line in file:
                line = line.strip()
                elements = line.split(split_char)
                elements[0] = int(elements[0]) / 10000000     # Time
                elements[2] = int(elements[2], 16)       # ascii (hex)
                elements[3] = int(elements[3], 16)            # scan code
                lst.append(elements)

        return lst


if __name__ == '__main__':
    base_folder = os.getcwd() + r"/TestFiles/1/"     
    Model( meta_data_file= base_folder+"random.txt", append_to_model= True ).train()
    cp.Info("A model object is created!")
    

#     # meta_data_file = r"E:\alireza\augusta\codes\recorder\dataset\hani/random.txt"
#     meta_data_file = r"E:\alireza\augusta\codes\recorder\dataset\hani/text.txt"
#
#     MatrixMaker(meta_data_file, c_sharp= True, add_pre_data=True).train()
