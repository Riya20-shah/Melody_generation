import json
import os

import tensorflow as tf
import music21 as m21
import numpy as np

# m21.configure.run()
# music 21 is package1 to convert the file into one data to another
# kern , MIDI , MusicXML  -> m21 -> kern , MIDI , ....
KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "Dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "Mapping.json"
SEQUENCE_LENGTH = 64
ACCEPTABLE_DURATIONS = [
    0.25,     #16th  note
    0.5,      #8th note
    0.75,
    1,        # quater note
    1.5,
    2,        # half note
    3,
    4         # whole note
]
def load_songs_in_kern(dataset_path):
    songs = []
    # go through all the files in dataset and load them with music21
    for path , subdir , files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
        return songs

def has_acceptable_durations(song,acceptable_duration):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_duration:
            return False
    return True

def transpose(song):
    # get key from the song

    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21

    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
        # print(key)
    # get the interval for transpostion   E.g. Bmaj --> Cmaj

    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic , m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))

    # transpose song by calculated interval

    transposed_song = song.transpose(interval)
    return transposed_song

def encode_song(song ,time_step=0.25):

    """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
       quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
       for representing notes/rests that are carried over into a new time step. Here's a sample encoding:
           ["r", "_", "60", "_", "_", "_", "72" "_"]
       :param song (m21 stream): Piece to encode
       :param time_step (float): Duration of each time step in quarter length
       :return:
       """

    encoded_song = []

    for event in song.flat.notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60

        # handle rests

        if isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note / rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        # print(event.duration.quarterLength)
        # print(steps)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)

            else:
                encoded_song.append("_")
            #cast encoded song to str

    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

def preprocess(dataset_path):
    #     load the folk songs
    print("songs Loading...")
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f"length of songs is {len(songs)}")

    # filter out songs that have non-acceptable duration
    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to cmaj/Amin
        song = transpose(song)

        # encode song with music time series represention
        encoded_song = encode_song(song)

        #     save songs to text file

        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as f:
            f.write(encoded_song)


def load(file_path):
    with open(file_path , "r") as fp:
        song = fp.read()
    return song
def create_single_file_dataset(dataset_path , file_dataset_path , sequence_length):

    """Generates a file collating all the encoded songs and adding new piece delimiters.
        :param dataset_path (str): Path to folder containing the encoded songs
        :param file_dataset_path (str): Path to file for saving songs in single file
        :param sequence_length (int): # of time steps to be considered for training
        :return songs (str): String containing all songs in dataset + delimiters
        """
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    # load encoded songsand delimeters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path , file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # remove empty space from the last character of string
    songs = songs[:-1]
    # save string that contain all dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs

def create_mapping(songs, mapping_path):
    """Creates a json file that maps the symbols in the song dataset onto integers
        :param songs(str): String with all songs
        :param mapping_path(str): Path where to save mapping
        :return:
        """

    mapping = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    # print(songs)
    # create mapping
    for i, symbol in enumerate(vocabulary):
        mapping[symbol] = i

    # save vocabulary to json file
    with open(mapping_path, "w") as fp:
        json.dump(mapping, fp, indent=4)

def convert_songs_to_int(songs):
    int_songs = []

    # load mapping
    with open(MAPPING_PATH , 'r') as f:
        mappings = json.load(f)

    # transform songs string to list

    songs = songs.split()

    # map song to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_sequences(sequence_length):
    # [11 ,12 ,13,14 , ...] --> input : [11,12] target: 13  , input : [12,13] target: 14
    """Create input and output data samples for training. Each sample is a sequence.
       :param sequence_length (int): Length of each sequence. With a quantisation at 16th notes, 64 notes equates to 4 bars
       :return inputs (ndarray): Training inputs
       :return targets (ndarray): Training targets
       """

    # load the song and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    # print(int_songs)
    inputs = []
    targets = []
    # generate the training sequences
    num_sequence = len(int_songs) - sequence_length
    for i in range(num_sequence):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append((int_songs[i+sequence_length]))
    # one-hot encoded the sequences

    # input size = [# num of sequences , sequence length , vocabulary size)
    vocabulary_size = len(set(int_songs))
    inputs = tf.keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"there are {len(inputs)} sequences")

    return inputs , targets

def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs , targets = generate_training_sequences(sequence_length=SEQUENCE_LENGTH)
    # print(inputs.shape)

if __name__ == "__main__":
    us = m21.environment.UserSettings()
    # us["musicxmlPath"] = "/home/scaledge-riya/.local/share/applications/org.musescore.MuseScore4portable.desktop"
    # us.delete()
    # songs = load_songs_in_kern(KERN_DATASET_PATH)
    # print(f"number of {len(songs)} songs is loaded .")

    # preprocess(KERN_DATASET_PATH)
    # songs = create_single_file_dataset(SAVE_DIR , SINGLE_FILE_DATASET , SEQUENCE_LENGTH)
    # create_mapping(songs , MAPPING_PATH)
    # main()
    print(us["musicxmlPath"])