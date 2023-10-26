import json

import music21 as m21
import numpy as np
import tensorflow as tf
from tensorflow import keras
from preprocessing import SEQUENCE_LENGTH , MAPPING_PATH

class MelodyGenerator:

    def __init__(self , model_path = "melody_generate_model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path , compile=False)

        with open(MAPPING_PATH , "r") as f:
            self._mappings = json.load(f)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self , seed , num_steps , max_sequence_length , temprature):
        """Generates a melody using the DL model and returns a midi file.
               :param seed (str): Melody seed with the notation used to encode the dataset   " 62 _ 63 _ _"
               :param num_steps (int): Number of steps to be generated
               :param max_sequence_len (int): Max number of steps in seed to be considered for generation
               :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
                   A number closer to 1 makes the generation more unpredictable.
               :return melody (list of str): List with symbols representing a melody
               """

        # create seed with start symbols

        seed = seed.split()

        melody = seed

        seed = self._start_symbols+seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one hot encoded seed
            onehot_seed = keras.utils.to_categorical(seed , num_classes=(len(self._mappings)))
            # it is generate the [max_sequence_length , number of symbol in the vocubulary ]
            # but we want in this shapes [1,max_sequence_length , number of symbol in the vocubulary]

            onehot_seed = onehot_seed[np.newaxis , ...]

            # make prediction

            probabilities = self.model.predict(onehot_seed)[0]

            # [0.1 ,0.2 ,0.1,0.6]  -> 1

            output_int = self._sample_with_temperature(probabilities , temprature)
            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k , v in self._mappings.items() if v == output_int][0]

            # check wether we are at the end of melody

            if output_symbol =='/':
                break

            # update melody
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self , probabilities , temperature):
        """Samples an index from a probability array reapplying softmax using temperature
                :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
                :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
                    A number closer to 1 makes the generation more unpredictable.
                :return index (int): Selected output symbol
                """
        predictions = np.log(probabilities) / temperature
        probabilities  = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))  # [0,1,2,3]

        index = np.random.choice(choices , p = probabilities)

        return index

    def save_melody(self , melody ,step_duration = 0.25 , format = "midi" , file_name = "mel.mid"):

        """Converts a melody into a MIDI file
               :param melody (list of str):
               :param min_duration (float): Duration of each time step in quarter length
               :param file_name (str): Name of midi file
               :return:
               """
        # create a music21 stream
        stream = m21.stream.Stream()
        start_symbol = None
        step_counter = 1

        # parse all the symbol in the melody and create note / rest object

        for i , symbol in enumerate(melody):

             # handle case in witch we have a not0e /rest

             if symbol != "_" or i+1 == len(melody):

                 # ensure we're dealing with note / rest beyond the first one
                 if start_symbol is not None:
                    quarter_length_duration = step_duration*step_counter   #0.25*4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarter_length_duration)

                    else:
                        m21_event = m21.note.Note(int(start_symbol),quarter_length = quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1
                 start_symbol = symbol

             # handle case in witch we have prolongation sign "_"
             else:
                step_counter+=1

        #write the m21 strams to mid files
        stream.write(format , file_name)

if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ _ _ _ _ 64 _ _ _ _ _ _ _ 67 _ _ _ _ _ _ _ _ _ _ _ 65"
    seed2 = "55 _ 60 _ 62 _ 64 _ 65 _ 67 _ _ _ 64 _ 72 _"
    melogy = mg.generate_melody(seed2,300,SEQUENCE_LENGTH, 0.4)
    print(melogy)
    mg.save_melody(melogy)