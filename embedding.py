from gensim.models.keyedvectors import KeyedVectors


class Embedding:
    def __init__(self, path):
        self.w2v = KeyedVectors.load(path)

    def call(self, tokens):
        """
        将传入的单词张量转换为词向量张量
        Args:
            tokens: 2D单词张量[['车主','说','好的'],['技师','走了']]
            size=[batch, timesteps]
        Returns:
            3D词向量张量[[[车主的词向量],[说的词向量]...][[技师词向量][走了词向量]]]
            size=[batch, timesteps, features]
        """
        matrix = []
        for seq in tokens:
            seq_tmp = []
            for token in seq:
                seq_tmp.append(self.w2v[token])
            matrix.append(seq_tmp)
        return matrix
