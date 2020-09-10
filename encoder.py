import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, enc_units, batch_sz, embedding):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_sz = batch_sz
        self.embedding = embedding
        self.gru = tf.keras.layers.GRU(units=enc_units, return_sequences=True)

    def call(self, x):
        x = self.embedding(x)
        hidden = self.initialize_hidden_state()
        # output 是encoder最终输出向量，用于decoder输入
        # states_hidden 是所有时间步的output拼接到一起，用于attention
        output, hidden_states = self.gru(x, initial_state=hidden)
        return output, hidden_states

    def init_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))