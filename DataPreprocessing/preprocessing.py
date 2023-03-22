
import os
import music21 as m21
import json
import tensorflow as tf
import numpy as np
KERN_DATA_PATH="C:/Users/ismai/OneDrive/Masaüstü/MusicGenerator/DataPreprocessing/polska"
ACCEPTABLE_DURATIONS=[
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]

SAVE_DIR="C:/Users/ismai/OneDrive/Masaüstü/MusicGenerator/DataPreprocessing/dataset"
SINGLE_FILE_DATASET="file_dataset"
SEQUENCE_LENGTH=64
MAPPING_PATH="mapping.json"



def load_songs(data_path):

    songs=[]
    #go to all the files and load them with music21
    for path,subdir ,files in os.walk(data_path):
        for file in files:
            if file[-3:]=="krn":
                song=m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    return songs


def has_accept_durat(songs,acceptable_durations):
    for note in songs.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
        
    return True

def transpose(song):
    # get key from the song

    parts=song.getElementsByClass(m21.stream.Part)
    measures_part0=parts[0].getElementsByClass(m21.stream.Measure)
    key=measures_part0[0][4]


    # estimate key using music21

    if not isinstance(key,m21.key.Key):
        key=song.analyze("key")
    print(key)
    #get interval for transposition

    if key.mode=="major":
        interval=m21.interval.Interval(key.tonic,m21.pitch.Pitch("C"))
    elif key.mode=="minor":
        interval=m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))



    # transpose song by calculated interval

    transposed_song=song.transpose(interval)

    return transposed_song

def encode_song(song,time_step=0.25):
    encoded_song=[]
    for event in song.flat.notesAndRests:

        if isinstance(event,m21.note.Note):
            symbol=event.pitch.midi

        elif isinstance(event,m21.note.Rest):
            symbol="r"
        
        #convert into time series
        steps=int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step==0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    #encode to string
    encoded_song=" ".join(map(str,encoded_song))
    
    return encoded_song


def preprocess(data_path):
    pass
    # load the songs


    print("Loading songs...")
    songs=load_songs(data_path)
    print(f"Loaded {len(songs)} songs.")

    for i,song in enumerate(songs):

        #filter out songs
        if not has_accept_durat(song,ACCEPTABLE_DURATIONS):
            continue


        #transpoze songs to cmaj/amin
        song=transpose(song)

        #encode songs with music time series representation
        encoded_song=encode_song(song)


        #save songs to text file
        save_path=os.path.join(SAVE_DIR,str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path,"r") as fp:
        song=fp.read()
    return song

def create_single_file(dataset_path,file_dataset_path,sequence_length):

    new_song_delimiter="/ " * sequence_length
    songs=""
    #load encoded song and add delimiters
    for path,_,files in os.walk(dataset_path):
        for file in files:
            file_path=os.path.join(path,file)
            song=load(file_path)
            songs=songs+song+" "+new_song_delimiter

    songs=songs[:-1]


    #save string
    with open(file_dataset_path,"w") as fp:
        fp.write(songs)
    
    return songs
    

def create_mapping(songs,mapping_path):

    mappings={}
    songs=songs.split()
    vocabulary=list(set(songs))

    for i,symbol in enumerate(vocabulary):
        mappings[symbol]=i
    
    #save as json
    with open(mapping_path,"w") as fp:
        json.dump(mappings,fp,indent=4)
    

def convert_songs_to_int(songs):
    int_songs=[]

    with open(MAPPING_PATH,"r") as fp:
        mappings=json.load(fp)

    songs=songs.split()

    for symbol in songs:
        int_songs.append(mappings[symbol])
    
    return int_songs

def generate_training_sequences(sequence_length):

    songs=load(SINGLE_FILE_DATASET)
    int_songs=convert_songs_to_int(songs)
    inputs=[]
    targets=[]
    num_sequences=len(int_songs)-sequence_length

    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    vocabulary_size=len(set(int_songs))
    inputs=tf.keras.utils.to_categorical(inputs,num_classes=vocabulary_size)
    targets=np.array(targets)

    return inputs,targets


def main():
    preprocess(KERN_DATA_PATH)
    songs=create_single_file(SAVE_DIR,SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    create_mapping(songs,MAPPING_PATH)
    
    

if __name__=="__main__":
    main()

    