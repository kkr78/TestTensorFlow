import pandas as pd
data = pd.read_csv("/home/artie/training/training-prod.csv")

label_buffer = data["intent"].values
text_buffer = data["text"].values
unique_labels = sorted(data["intent"].unique())
print(set(label_buffer))
print(len(label_buffer))


############################################################  
######           Encode Labels and Text               ######
############################################################  

import keras
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from smart_open import smart_open
from vectoria import CharacterTrigramEmbedding, WordEmbedding, FastTextEmbedding

import pandas as pd
from keras.utils import to_categorical

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(label_buffer)
#print(list(set(labels)))

one_hot_labels =  to_categorical(labels)

#print(one_hot_labels[:10])

embedding = CharacterTrigramEmbedding()
#embedding = WordEmbedding()
#embedding = FastTextEmbedding()
print("embedding>>",embedding)
texts = embedding.sequencer.transform(text_buffer)
#print("text>>",texts[:10])

# one hot encode the labels
targets = one_hot_labels

# now set up sequencing and embedding
sources = texts

HIDDEN = 128


print("Embedding completed>>",embedding)

#############################################################  
######                  Recurrent/LSTM (2)             ######
#############################################################  
model_name='recurrent-lstm'
class Reverse(keras.layers.Layer):
    """
    A custom keras layer to reverse a tensor.
    """

    def call(self, tensor):
        """
        Use the backed to reverse.
        """
        return keras.backend.reverse(tensor, 1)

    def compute_output_shape(self, input_shape):
        """
        No change in shape.
        """
        return input_shape

inputs = keras.layers.Input(shape=(embedding.maxlen,))
# embedding to turn ngram identifiers dense
embedded = embedding.build_model()(inputs)
stack = keras.layers.Conv1D(HIDDEN, 3, activation='relu')(embedded)
stack = keras.layers.MaxPooling1D(3)(stack)
stack = keras.layers.Conv1D(HIDDEN, 3, activation='relu')(stack)
stack = keras.layers.MaxPooling1D(3)(stack)
stack = keras.layers.Conv1D(HIDDEN, 3, activation='relu')(stack)
stack = keras.layers.MaxPooling1D(3)(stack)
stack = keras.layers.Dropout(0.5)(stack)
# recurrent layer -- read the word like structures in time series order
# note this is GPU only, and keras
# '>=2.0.9, it is shocking slow otherwise
recurrent_forward = keras.layers.CuDNNLSTM(HIDDEN)(stack)
recurrent_backward = keras.layers.CuDNNLSTM(HIDDEN)(Reverse()(stack))
#recurrent_forward = keras.layers.LSTM(HIDDEN)(stack)
#recurrent_backward = keras.layers.LSTM(HIDDEN)(Reverse()(stack))
recurrent = keras.layers.Concatenate()([recurrent_forward, recurrent_backward])
recurrent = keras.layers.Dropout(0.5)(recurrent)
# dense before final output
stack = keras.layers.Dense(HIDDEN, activation='relu')(recurrent)
stack = keras.layers.Dropout(0.5)(stack)
stack = keras.layers.Dense(HIDDEN, activation='relu')(stack)
stack = keras.layers.Dropout(0.5)(stack)
# softmax on two classes -- which map to our 0, 1 one hots
outputs = keras.layers.Dense(13, activation='softmax')(stack)
model = keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()
model.fit(
    x=sources,
    y=targets,
    validation_split=0.05,
    batch_size=100,
    epochs=20


############################################################  
######                  SAVE MODEL LOCAL              ######
############################################################  
import datetime as dt

MODEL_FILE="model_"+model_name+"_"+dt.datetime.now().strftime("%Y%m%d%H%M%S")+".h5"
print(MODEL_FILE)
model.save(MODEL_FILE)

# Saved model stats
import os
stats = os.stat(MODEL_FILE)
print(stats)
print(stats.st_size)
