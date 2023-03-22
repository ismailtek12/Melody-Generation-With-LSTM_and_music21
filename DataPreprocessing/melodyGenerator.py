import tensorflow as tf
import json
from preprocessing import SEQUENCE_LENGTH,MAPPING_PATH
import numpy as np
import h5py
class MelodyGenerator:

    def __init__(self,model_path="C:/Users/ismai/OneDrive/Masaüstü/MusicGenerator/model.h5"):
        self.model_path=model_path
        self.model=tf.keras.models.load_model(model_path)
        
        with open(MAPPING_PATH,"r") as fp:
            self._mappings=json.load(fp)
        
        self._start_symbols=["/"]*SEQUENCE_LENGTH
    

    def generate_melody(self,seed,num_steps,max_sequence_length,temperature):

        #create seed
        seed=seed.split()
        melody=seed
        seed=self._start_symbols+seed

        #convert seed to int
        seed=[self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            seed=seed[-max_sequence_length]
            one_hot_seed=tf.keras.utils.to_categorical(seed,num_classes=len(self._mappings))
            one_hot_seed=one_hot_seed[np.newaxis,...]


            #make a prediction

            probabilities=self.model.predict(one_hot_seed)[0]
            output_int=self._sample_with_temperature(probabilities,temperature)

            seed.append(output_int)

            output_symbol=[k for k,v in self._mappings.items() if v==output_int]

            if output_symbol=="/":
                break
            melody.append(output_symbol)
        return melody


    def _sample_with_temperature(self,probabilities,temperature):

        predictions=np.log(probabilities)/temperature
        probabilities=np.log(predictions)/np.sum(np.exp(predictions))

        choices=range(len(probabilities))
        index=np.random.choice(choices,p=probabilities)

        return index


if __name__=="__main__":
    mg=MelodyGenerator()
    seed="55 _ 60 _ 60 _ 64 _ 60 60 60 _ 59 _ 59"
    melody=mg.generate_melody(seed,500,SEQUENCE_LENGTH,0.7)
    print(melody)