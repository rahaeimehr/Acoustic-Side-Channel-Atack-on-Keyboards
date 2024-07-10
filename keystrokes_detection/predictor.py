import logging
import pandas as pd

# logging.basicConfig(handlers=[
#     logging.FileHandler("log.txt"),
#     logging.StreamHandler()
# ], level=logging.INFO)


class KeystrokesPredictor:
    def __init__(self, model_address):
        self.delta_time = 0
        self.delta_time_factor = 0
        self.time_factor = 1.05
        self.file_address = model_address + f'./model.xlsx'
        self.sheet_num = 10
        self.sheets_data = []
        self.sheets_data_analyzed = []
        self.read_trained_model()

    def read_trained_model(self):
        self.sheets_data = pd.read_excel(self.file_address, sheet_name=f'data', header=0).to_numpy()
        self.sheets_data_analyzed = pd.read_excel(self.file_address, sheet_name=f'analysis', header=0).to_numpy()

    def keystrokes_finder(self, time_min, time_max, analyzed=True):
        predicted_keys = []
        if analyzed:
            trained_data = self.sheets_data_analyzed
        else:
            trained_data = self.sheets_data

        for row in trained_data:
            # print(row[2])
            if (row[2] >= time_min) & (row[2] <= time_max):
                predicted_keys.append([row[0], row[1]])

        if not predicted_keys:
            # print("nothing---------\n\n")
            predicted_keys = [['None', 'None']]
        return predicted_keys


# if __name__ == '__main__':
#     predictor = KeystrokesPredictor()
#     print(predictor.keystrokes_finder(0.375887, 0.4))
