import tensorflow as tf

class TransformerBuilder:
    def __init__(self):
        self.N = 1
        self.num_heads = 4
        self.dropout_rate = 0.1
        self.n_units = 128  
        self.embed_size = 128
    def build(self, encoder_in, decoder_in):


        Z = encoder_in
        for _ in range(self.N):
            skip = Z
            attn_layer = tf.keras.layers.MultiHeadAttention(
                num_heads = self.num_heads,
                key_dim = self.embed_size,
                dropout = self.dropout_rate
            )
            Z = attn_layer(Z, value = Z)
            Z = tf.keras.layers.LayerNormalization()(Z+ skip)
            skip = Z
            Z = tf.keras.layers.Dense(self.n_units, activation="relu")(Z)
            Z = tf.keras.layers.Dense(self.embed_size)(Z)
            Z = tf.keras.layers.Dropout(self.dropout_rate)(Z)
            Z = tf.keras.layers.LayerNormalization()(Z+ skip)


        encoder_outputs = Z 
        Z = decoder_in
        for _ in range(self.N):
            skip = Z
            attn_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_size, 
                dropout=self.dropout_rate
            )
            Z = attn_layer(Z,value=Z, use_causal_mask = True)
            skip = Z
            attn_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=self.embed_size, 
                dropout=self.dropout_rate
            )
            Z = attn_layer(Z, value=encoder_outputs)
            Z = tf.keras.layers.LayerNormalization()(Z+ skip)
            skip = Z
            Z = tf.keras.layers.Dense(self.n_units, activation="relu")(Z)
            Z = tf.keras.layers.Dense(self.embed_size)(Z)
            Z = tf.keras.layers.Dropout(self.dropout_rate)(Z)
            Z = tf.keras.layers.LayerNormalization()(Z+ skip)
        return Z


