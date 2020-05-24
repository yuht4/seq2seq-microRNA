import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import copy

from auix import *

tf.config.experimental_run_functions_eagerly(True)

def encode_seq(seq):

    dict_seq = {'<PAS>':0, 'A':1, 'C':2, 'G':3,'U':4, '<start>':5, '<stop>':6}
    data = []

    for val in seq:
        if val in dict_seq.keys():
            data.append(dict_seq[val]);
        else:
            print("error")

    data = [5] + data + [6];
    number = len(data)
    data = data + [0] * (31 - number)

    return data; ### without start and stop


def load_dataset_batched():

    data = pd.read_csv("clean_data_suffeled.csv")

    mRNA = np.array(data['mRNA'])
    microRNA = np.array(data['microRNA'])

    mRNA = np.array([encode_seq(x) for x in mRNA])
    microRNA = np.array([encode_seq(x) for x in microRNA])

    microRNA_inp =  microRNA[:, :-1]
    microRNA_real = microRNA[:, 1:]

    return mRNA, microRNA_inp, microRNA_real;


def main():

    mRNA, microRNA_inp, microRNA_real =  load_dataset_batched();
    
    model = getModel();

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 10)
    model_check = tf.keras.callbacks.ModelCheckpoint(filepath = "./transformer_embed.h5", monitor = 'val_loss', save_best_only= True, save_weights_only=True)
    
    model.fit({'inputs_seq':mRNA[:18261], 'targets_input' : microRNA_inp[:18261] }, microRNA_real[:18261],
            batch_size = 64,
            epochs = 100,
            validation_split = 0.15, callbacks = [early_stopping, model_check]
            );

    # return model;


def getModel():

    inputs_seq = tf.keras.Input(shape=(None,),name='inputs_seq');
    targets_input = tf.keras.Input(shape=(None,),name='targets_input');
    
    num_layers = 3
    d_model = 128
    dff = 128
    num_heads = 8

    input_vocab_size = 7
    target_vocab_size = 7
    dropout_rate = 0.1

    transformer = Transformer(num_layers, d_model, num_heads, dff,input_vocab_size, target_vocab_size, 
                          pe_input=35, pe_target=35,rate=dropout_rate)

    decoder_output, _ = transformer(inputs_seq, targets_input)   

    model = tf.keras.Model(inputs=[inputs_seq,targets_input], outputs= decoder_output);

    model.compile(
        optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98,epsilon=1e-9), 
        loss = loss_function,
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    );

    return model;


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



def evaluate(seq, model):

    dict_seq = {'<PAS>':0, 'A':1, 'C':2, 'G':3,'U':4, '<start>':5, '<stop>':6}
    encoder_input = tf.expand_dims(encode_seq(seq), 0)

    output = tf.expand_dims([5], 0)
    
    for i in range(30):

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = model.predict({'inputs_seq':encoder_input, 'targets_input':output})

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == 6:
            return tf.squeeze(output, axis=0)

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def translate(sentence, model):
    result = evaluate(sentence, model)
    dict_seq = {0 : '<PAS>', 1 : 'A', 2 : 'C', 3 : 'G',4 : 'U', 5 : '<start>', 6 : '<stop>'}

    pre_str = ""

    for i in result.numpy()[1:]:
        pre_str += dict_seq[i]
    return pre_str;



if __name__ == '__main__':
   main()
