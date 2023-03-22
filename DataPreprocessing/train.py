from preprocessing import generate_training_sequences,SEQUENCE_LENGTH
import tensorflow as tf
OUTPUT_UNITS=23
LOSS="sparse_categorical_crossentropy"
LEARNING_RATE=0.001
NUM_UNITS=[256]
EPOCHS=20
BATCH_SIZE=64
SAVE_MODEL_PATH="model.h5"

def build_model(output_units,num_units,loss,learning_rate):
    input=tf.keras.layers.Input(shape=(None, output_units))
    X=tf.keras.layers.LSTM(num_units[0])(input)
    X=tf.keras.layers.Dropout(0.2)(X)
    output=tf.keras.layers.Dense(output_units,activation="softmax")(X)
    model=tf.keras.Model(input,output)

    model.compile(loss=loss,optimizer=tf.keras.optimizers.Adam(lr=learning_rate),metrics=["accuracy"])

    model.summary()

    return model

def train(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):
    inputs,targets=generate_training_sequences(SEQUENCE_LENGTH)
    model=build_model(output_units,num_units,loss,learning_rate)

    model.fit(inputs,targets,epochs=EPOCHS,batch_size=BATCH_SIZE)

    tf.keras.models.save_model(model,"model.h5")
    



if __name__=="__main__":
    train()