"""
Written by Alireza Taheri ( March 2022 )

"""
from keylogger import Keylogger
from sound_recorder import record
import os
import multiprocessing as mp


def get_last_file_name(directory):
    os.makedirs(directory, exist_ok=True)
    i = 1
    while os.path.isdir(f'{directory}/{i}'):
        i += 1
    os.makedirs(f'{directory}/{i}', exist_ok=True)
    return f"{directory}/{i}"


if __name__ == '__main__':
    mp.freeze_support()
    print("*** Keylogger app is running ***")
    manager = mp.Manager()

    file_names = get_last_file_name(directory='./datasets')
    print("File address:", file_names)
    keywords = {'file_name': file_names}
    keylogger_process = mp.Process(target=Keylogger, kwargs=keywords)
    sound_recorder = mp.Process(target=record, kwargs=keywords)
    keylogger_process.start()
    sound_recorder.start()

    sound_recorder.join()
    keylogger_process.join()


    # from wave_spliter import WordSpliter
    # wave_spliter = WordSpliter(n_seconds_of_silence=7, file_address=file_names,
    #                            remove_from_start_time=1, factor=3).splitter()

