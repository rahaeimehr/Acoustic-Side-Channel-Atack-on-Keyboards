from keystrokes_detector_energy import KeyDdetector
from train import MatrixMaker
from word_spliter import WordSpliter
from word_spliter import AudioSpliter
from hex_to_ascii import main_converter
import data_structures as ds
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
dataFolder = "1"    
folder = current_file_path + r"/TestFiles/" + dataFolder + "/"
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
    
    noise_sample = ds.AudioFile(folder + r"random.wav", folder + r"random.txt")
    noise_sample.plot_keystrokes_on_waveform()
    print(noise_sample.sound_data)
    audio_to_split = ds.AudioFile(folder + r"words.wav" , folder + r"words.txt")
    audio_to_split.plot_keystrokes_on_waveform()
    print("Noise First key:", noise_sample.get_nth_keystrokePressTime(0), noise_sample.get_nth_keystrokePressTime(1))
    print("Audio First key:", noise_sample.get_nth_keystrokePressTime(0), noise_sample.get_nth_keystrokePressTime(1))

    audio_spliter = AudioSpliter(audio_file=audio_to_split)
    audio_spliter.set_noise_sample(noise_sample)
    
    
    audio_spliter.test()
    
if not b:
    print("\n\nSPLIT WORDS WAVE:")
    file_address = folder

    n_seconds_of_silence = 0
    remove_from_start_time = 0
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
    parent_path = folder + r'words/'
    file = parent_path + r'word_4.wav'
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


