import tensorflow as tf


class Decoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, batch_sz, vocab_sz, embedding):
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.vocab_sz = vocab_sz
        self.embedding = embedding
        self.gru = tf.keras.layers.GRU(units=dec_units, return_sequences=True)
        self.fc = tf.keras.layers.Dense(units=vocab_sz, activation=None)

    def call(self, dec_inp, hidden_state):
        x = self.embedding(dec_inp)
        hidden = hidden_state
        # states: 各个timestep的输出/隐层状态,size=[batch,seq_len,dec_units]
        output, states = self.gru(x, initial_state=hidden, return_sequences=True)
        output = self.fc(states)
        probs = tf.nn.softmax(output)

        return probs

