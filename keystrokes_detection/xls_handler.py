import xlsxwriter
import pandas as pd

def write_matrix_to_file(data, file_address, wordsheet_title="data", column_titles=[]):
    workbook = xlsxwriter.Workbook(file_address)
    worksheet = workbook.add_worksheet(wordsheet_title)
    for index, title in enumerate(column_titles):
        worksheet.write(0, index, title)

    for row_index, row in enumerate(data):
        for col_index, column in enumerate(row):
            worksheet.write(row_index+1, col_index, column)
    workbook.close()

def read_excel_data(file_name, sheet_name="data"):
    df = pd.read_excel(file_name, sheet_name=sheet_name)   
    # Convert the DataFrame to a 2D array
    data = df.values.tolist()
    return data


