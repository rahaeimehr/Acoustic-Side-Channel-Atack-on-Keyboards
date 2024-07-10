import xlsxwriter


def split_line(filename, split_char=','):
    # print(filename)
    lst = list()
    with open(filename) as file:
        for line in file:
            line = line.strip()
            elements = line.split(split_char)
            # print(elements)
            elements[0] = int(elements[0]) / 10000000  # Time
            elements[2] = chr(int(elements[2], 16))  # ascii (hex)
            elements[3] = int(elements[3], 16)  # scan code
            # print("elements:", elements)
            lst.append(elements)
    return lst


def hex_to_ascii(dataset_csv_file_address):
    keystrokes = []
    up_times = []
    down_time = []
    data = []
    df_list = split_line(dataset_csv_file_address)
    for i in range(int((len(df_list)) / 2)):
        keystrokes.append(df_list[2 * i][2])
        up_times.append(df_list[2 * i + 1][0])
        down_time.append(df_list[2 * i][0])

    for i in range(len(keystrokes) - 1):
        delta_average = ((down_time[i + 1] + up_times[i + 1]) / 2) - ((down_time[i] + up_times[i]) / 2)
        delta_down_to_down = (down_time[i + 1] - down_time[i])
        delta_up_to_up = (up_times[i + 1] - up_times[i])
        data.append([keystrokes[i], keystrokes[i + 1], delta_average, delta_down_to_down, delta_up_to_up])
    return data


def write_matrix_to_file(data, file_address):
    workbook = xlsxwriter.Workbook(f'{file_address}/all_words.xlsx')
    worksheet = workbook.add_worksheet(f"data")
    worksheet.write(0, 0, 'First_Key')
    worksheet.write(0, 1, 'Second_Key')
    worksheet.write(0, 2, 'Delta_Time_up_to_up')
    worksheet.write(0, 3, 'Delta_Time_down_to_down')
    worksheet.write(0, 4, 'Delta_Time_average')

    for index, row in enumerate(data):
        worksheet.write(index + 1, 0, row[0])
        worksheet.write(index + 1, 1, row[1])
        worksheet.write(index + 1, 2, row[2])
        worksheet.write(index + 1, 3, row[3])
        worksheet.write(index + 1, 4, row[4])
    workbook.close()


def main_converter(file_address):
    dataset_csv_file_address = file_address + r"/words.txt"
    data = hex_to_ascii(dataset_csv_file_address)
    write_matrix_to_file(data, file_address)
