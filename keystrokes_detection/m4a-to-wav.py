from pydub import AudioSegment

# Define input and output file paths
path = r"E:\alireza\augusta\codes\recorder-english\dataset\new/"
input_file = path + "new.m4a"
output_file = path + "new-words.wav"


track = AudioSegment.from_file(input_file, format='m4a')
file_handle = track.export(output_file, format='wav')

