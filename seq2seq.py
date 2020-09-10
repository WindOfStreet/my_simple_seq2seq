import os
import sys
import tensorflow as tf

from seq2seq.encoder import Encoder
from seq2seq.decoder import Decoder
from tools.utils import load_w2v_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class SeqToSeq(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
        self.embedding = load_w2v_model(BASE_DIR+"word2vec/output/gensim_wv_model.wv.wv")
        self.encoder = Encoder(params['enc_units'],
                               params['batch_sz'],
                               self.embedding)
        self.decoder = Decoder(params['dec_units'],
                               params['batch_sz'],
                               params['vocab_sz'],
                               self.embedding)

    def call(self, enc_input, dec_input):
        """
        运行encoder和decoder
        Args:
            enc_input: 2d-tensor，输入语料序列，size=[batch,seq_len]
            dec_input: decoder初始输入，size=[batch,1]

        Returns:
            生成的序列？
        """
        enc_output, enc_hidden_states = self.encoder(enc_input)
        probs = self.decoder(dec_input, enc_output)


