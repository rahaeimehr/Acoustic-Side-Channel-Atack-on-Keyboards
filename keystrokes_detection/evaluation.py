import os
import glob
import xlsxwriter
from keystrokes_detector_energy import KeyDdetector
from train import MatrixMaker
from word_spliter import WordSpliter
from hex_to_ascii import main_converter
import pandas as pd
                    
def write_to_xlsx_file(all_dict):
    workbook = xlsxwriter.Workbook(f'{datasets_folder_path}/result.xlsx')
    worksheet = workbook.add_worksheet(f"all_results")
    worksheet.write(0, 0, 'Dataset Name')
    worksheet.write(0, 1, 'Word Number')
    worksheet.write(0, 2, 'GT')
    worksheet.write(0, 3, 'Pred')
    worksheet.write(0, 4, 'Score')
    worksheet.write(0, 5, 'Score_n')

    data_index = 0
    for res_dict in all_dict:    
          
        worksheet.write(data_index + 1, 0, res_dict['dataset_name'])
        worksheet.write(data_index + 1, 1, res_dict['word_number'])
        worksheet.write(data_index + 1, 2, res_dict['gt'])
        worksheet.write(data_index + 1, 3, str(res_dict['pred']))
        worksheet.write(data_index + 1, 4, res_dict['score'])
        worksheet.write(data_index + 1, 5, res_dict['score_n'])
        
        data_index += 1

    workbook.close()
    
def evaluation(folder_path):
        
    all_dict = []
    for file in os.listdir(folder_path):
        folder = os.path.join(folder_path, file)
        if os.path.isdir(folder):
            print(folder)          
            print(f"Current dataset: {folder}")
            
            print("TRAIN PHASE:")
            dataset_csv_file_address_1 = os.path.join(folder , r"random.txt")
            dataset_csv_file_address_2 = os.path.join(folder , r"text.txt")
            dataset_csv_file_address_3 = os.path.join(folder , r"words.txt")

            print("TRAIN 1:")
            MatrixMaker(dataset_csv_file_address_1, output_address=folder, c_sharp=True, add_pre_data=False).train()
            print("TRAIN 2:")
            MatrixMaker(dataset_csv_file_address_2, output_address=folder, c_sharp=True, add_pre_data=True).train()
            print("TRAIN 3:")
            MatrixMaker(dataset_csv_file_address_3, output_address=folder, c_sharp=True, add_pre_data=True).train()
            
            print("SPLIT WORDS WAVE:")
            file_address = folder

            n_seconds_of_silence = 4
            remove_from_start_time = 1
            factor = 1
            c_sharp = True
            try:
                wave_spliter = WordSpliter(n_seconds_of_silence=n_seconds_of_silence,
                                        file_address=file_address, c_sharp=c_sharp,
                                        reduce_noise=True,
                                        remove_from_start_time=remove_from_start_time, factor=factor).splitter()

                main_converter(folder)
            except Exception as e:
                print(e)
                continue
            
            print("PREDICT:")
            parent_path = os.path.join(folder , r'words')
            i = 0
            flag = 0
            score = 0
            score_n = 0
            for path in glob.glob(f'{parent_path}/*.wav'):
                
                print("\nWAV file:", path)
                xlsx_file = path.split('.')[0]+ '.xlsx'
                print("xlsx file:", xlsx_file)
                
                if i != 0:
                    try:
                        output_dir = path.split('.')[0] + '/'
                        print("Output folder:", output_dir)
                        
                        
                        # readeing te xlsx file fo find the number of keystrokes in each word:
                        dataframe = pd.read_excel(xlsx_file)
                        test_word = ''
                        for ind in dataframe.index:
                            # print(dataframe['key'][ind])
                            test_word += dataframe['key'][ind]
                        print("Groundtruth_word:", test_word)
                        if len(test_word) > 7:
                            print("********** Wow BIG WORD **********")
                            continue
                        plot = False
                        analyzed = False
                        delta_time_min_factor = 0.95
                        delta_time_max_factor = 1.05
                        num_desired_keystrokes = len(dataframe)

                    
                        output_words = KeyDdetector(file_path=path, output_dir=output_dir, plot=plot, num_desired_keystrokes=num_desired_keystrokes,
                                                        analyzed=analyzed, delta_time_min_factor=delta_time_min_factor,
                                                        delta_time_max_factor=delta_time_max_factor, model_address=folder).key_detection()
                        print ("predicted_words numbers", len(output_words))
                        if test_word in output_words:
                            print("Successfully Detected!\n\n")
                            flag = 1
                        else:
                            flag = 0
                            
                        score += flag
                        score_n += flag/len(output_words)  
                        print(i, score, score_n)
                        # res_dict = {"dataset_name":folder.split("/")[1], "word_number":path.split("\\")[-1],\
                        #             "gt":test_word, "pred":output_words, "score":flag, "score_n":flag/len(output_words) }
                        x = flag/len(output_words)
                        y = path.split("\\")[-1] # 
                        z = str(output_words).replace(",", "-").replace("\'", "").replace("[", "").replace("]", "")
                        res_dict = f'{str(folder.split("/")[1])}, {y}, {flag}, {x}, {test_word}, {z} \n'

                        print("Final Score:", score/i)
                        print("Final Score N:", score_n / i)
                        i += 1
                        with open(f'{datasets_folder_path}/result.csv', 'a') as f:
                            f.write(res_dict)
                    
                    except Exception as e:
                        print(e)
                else:
                    i +=1          
            #     if i == 5:  
            #         break
            # break
    # write_to_xlsx_file(all_dict)
    
    
    
    
datasets_folder_path = r'C:\Users\ataheritajar\Box\dataset/'
evaluation(datasets_folder_path)





