import tensorflow as tf
import numpy as np
class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_length, embed_size, dtype = tf.float32, **kwars):
        super().__init__(dtype=dtype, **kwars)
        p,i  = np.meshgrid(np.arange(max_length),
                          2 * np.arange(embed_size // 2))
        pos_emb = np.empty((1, max_length, embed_size))                 

        pos_emb[0, :, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T
        self.pos_encodings = tf.constant(pos_emb.astype(self.dtype))
        self.supports_masking = True
    def call(self, inputs):
        batch_max_length = tf.shape(inputs)[1]
        fillted_pos_emb = self.pos_encodings[:, :batch_max_length]
        return inputs + fillted_pos_emb

      
        
