import keras
import os
from preprocessing import generate_training_sequences, SEQUENCE_LENGTH
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

OUTPUT_UNIT = 34    # num of output class (that is number of vocabulary in mapping  json )
NUM_UNITS = [256]   # internal layer of layers (hidden layer)
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 1
SAVE_MODEL_PATH = "melody_generate_model.h5"
def build_model(output_unit ,num_unit ,loss , learning_rate):
    # create model architecture
    input_ = tf.keras.layers.Input(shape=(None , output_unit))
    x = tf.keras.layers.LSTM(num_unit[0])(input_)
    x = tf.keras.layers.Dropout(0.2)(x)

    output = tf.keras.layers.Dense(output_unit , activation="softmax")(x)

    model = keras.Model(input_ , output)

    # compile model
    model.compile(loss=loss ,
                  optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate) ,
                  metrics=["accuracy"])

    model.summary()

    return model

def train(output_unit = OUTPUT_UNIT ,num_unit = NUM_UNITS ,loss = LOSS , learning_rate = LEARNING_RATE):
    # generate the training sequences

    inputs , targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network

    model = build_model(output_unit ,num_unit , loss , learning_rate)

    # train the model

    model.fit(inputs , targets , epochs=EPOCHS , batch_size=BATCH_SIZE)

    # save the model

    model.save(SAVE_MODEL_PATH)



if __name__ == "__main__":
    train()
