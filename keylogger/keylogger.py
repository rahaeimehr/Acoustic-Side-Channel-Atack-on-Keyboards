"""
Written by Alireza Taheri ( March 2022 )

"""
from time import time

# Python code for keylogger to be used in windows
import win32console
import pythoncom, pyWinhook
import sys
from datetime import datetime
from database_manager import DbHandler
# import xlsxwriter
import csv


class Keylogger:
    
    def __init__(self, file_name):
        self.stop_time = None
        self.file_name = file_name
        self.db_data = []
        self.number_of_key = 0
        self.win = win32console.GetConsoleWindow()
        self.txt_file = f"{self.file_name}/main.txt"
        self.db_file = f"{self.file_name}/main.db"
        # print("TEXT FILE NAMES:", self.txt_file)
        # print("DATABASE FILE NAMES:", self.db_file)
        self.db = DbHandler(self.db_file)
        self.time_up = None
        self.time_down = None
        self.is_first_key = True
        self.first_time_stamp = datetime.timestamp(datetime.now())
        print(self.first_time_stamp)
        self.run()

    def keyboard_event_down(self, event):
        self.time_down = datetime.timestamp(datetime.now()) - self.first_time_stamp
        return 1

    def keyboard_event_up(self, event):

        # print('MessageName: %s' % event.MessageName)
        # print('Message: %s' % event.Message)
        print('Time: %s' % event.Time)
        # print('Window: %s' % event.Window)
        # print('WindowName: %s' % event.WindowName)
        print('Ascii: %s' % event.Ascii, chr(event.Ascii))
        print('Key: %s' % event.Key)
        print('KeyID: %s' % event.KeyID)
        # print('ScanCode: %s' % event.ScanCode)
        # print('Extended: %s' % event.Extended)
        # print('Injected: %s' % event.Injected)
        # print('Alt %s' % event.Alt)
        # print('Transition %s' % event.Transition)
        # print('---')

        self.time_up = datetime.timestamp(datetime.now()) - self.first_time_stamp
        if event.Ascii == 27:
            # Write data to file:
            self.stop_time = datetime.timestamp(datetime.now()) - self.first_time_stamp
            self.read_and_write_to_file_and_db()
            print(f"time of duration: {datetime.timestamp(datetime.now()) - self.first_time_stamp}")

            print("Total number of pressed keys: ", self.number_of_key)
            # Exit
            sys.exit(1)

        if event.Ascii != 0 or event.Ascii != 8:
            self.db_data.append((self.time_down, self.time_up, event.Ascii, chr(event.Ascii), event.Key, event.KeyID))

        self.number_of_key += 1

        return 1

    def read_and_write_to_file_and_db(self, ):
        # open output.txt to read current keystrokes
        f = open(self.txt_file, 'a')
        # open output.txt to write current + new keystrokes
        for data in self.db_data:
            keylogs = f"\n{str(data[0])}, {str(data[1])}: {str(data[2])}"
            f.write(keylogs)
        f.close()
        # insert all data to database:
        self.db.db_insert(self.db_data)
        self.output()

    def run(self, ):
        # create a hook manager object
        hm = pyWinhook.HookManager()
        hm.KeyDown = self.keyboard_event_down
        hm.KeyUp = self.keyboard_event_up
        # set the hook
        hm.HookKeyboard()
        # wait forever
        pythoncom.PumpMessages()

    def output(self, ):

        with open(f'{self.file_name}/main.csv', 'w', newline='') as csvfile:
            fieldnames = ['Down_timestamp', 'Up_timestamp', 'Ascii', 'Chr', 'Key', 'KeyID', 'wave_start', 'stop_time']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            writer.writerow({'wave_start': self.first_time_stamp, 'stop_time': self.stop_time})

            for down_time, up_time, key_ascii, key_chr, key, key_id in self.db_data:
                writer.writerow({'Down_timestamp': down_time, 'Up_timestamp': up_time, 'Ascii': key_ascii,
                                 'Chr': key_chr, 'Key': key, 'KeyID': key_id})
