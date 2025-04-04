from keystrokes_detector_energy import KeyDdetector
from train import MatrixMaker
from word_spliter import WordSpliter
from hex_to_ascii import main_converter
import os
import helper

a = True
#a = False

b = True
#b = False

c = True
#c = False


# # TRAIN PHASE:
current_file_path = os.getcwd()
errors =[]

#for dataFolder in helper.get_folders(current_file_path+"/TestFiles"):
dataFolder = "638754091389248604943"    
folder = current_file_path + r"/dataset2/" + dataFolder + "/"
print("\n"*5 + folder)
if a:
    print("\n\nTRAIN PHASE:")
    dataset_csv_file_address_1 = folder + r"random.txt"
    dataset_csv_file_address_2 = folder + r"text.txt"
    dataset_csv_file_address_3 = folder + r"words.txt"
    try:
        print("\nTRAIN 1:")
        MatrixMaker(dataset_csv_file_address_1, output_address=folder, c_sharp=True, add_pre_data=False).train()
        print("\nTRAIN 2:")
        MatrixMaker(dataset_csv_file_address_2, output_address=folder, c_sharp=True, add_pre_data=True).train()
        print("\nTRAIN 3:")
        MatrixMaker(dataset_csv_file_address_3, output_address=folder, c_sharp=True, add_pre_data=True).train()
    except Exception as e:
        # Catches any other exceptions
        errors.append(e)
        print(f"An error occurred: {e}")

# SPLIT WORDS WAVE:
if b:
    print("\n\nSPLIT WORDS WAVE:")
    file_address = folder

    n_seconds_of_silence = 1
    remove_from_start_time = 1
    factor = 1
    c_sharp = True
    wave_spliter = WordSpliter(n_seconds_of_silence=n_seconds_of_silence,
                            file_address=file_address, c_sharp=c_sharp,
                            reduce_noise=True,
                            remove_from_start_time=remove_from_start_time, factor=factor).splitter()

    main_converter(folder)

if c:
    # PREDICT:
    print("\n\nPREDICT:")
    parent_path = folder + r'/words/'
    file = parent_path + r'word_34.wav'
    output_dir = parent_path + r'keystrokes_words_1/'

    plot = True
    analyzed = False
    delta_time_min_factor = 0.95
    delta_time_max_factor = 1.05
    num_desired_keystrokes = None

    KeyDdetector(file_path=file, output_dir=output_dir, plot=plot, num_desired_keystrokes=num_desired_keystrokes,
                analyzed=analyzed, delta_time_min_factor=delta_time_min_factor,
                delta_time_max_factor=delta_time_max_factor, model_address=folder).key_detection()


if len(errors)>0:
    print("-"*20 + " ERRORS "+ "-"*20)
    print(errors)
print("Done " + "."*40 + "\n"*5)


