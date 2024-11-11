import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Add, Activation, AveragePooling2D, BatchNormalization
import tensorflow_addons as tfa

def base_cnn(inputs):
    x1 = conv_block(inputs, 16)
    pool_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x1)

    x2 = conv_block(pool_1, 32)
    pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x2)

    x3 = conv_block(pool_2, 64)
    pool_3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x3)

    x4 = conv_block(pool_3, 128)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x4)

    return x
#---------------------------------------------------------------
def conv_block(inputs, num_filters=None):
    x1 = tf.keras.layers.SeparableConv2D(filters=num_filters, kernel_size=(3, 3),  padding='same', activation='relu')(inputs)
    x2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), dilation_rate=2, activation='relu', padding='same')(inputs)

    x = Concatenate()([x1, x2])
    x = BatchNormalization()(x)
    return x
#----------------------------------------------------------------
def mlp(x, mlp_dim, dim, dropout_rate=0.1):
    x = L.Dense(mlp_dim, activation="swish")(x)
    x = L.Dropout(dropout_rate)(x)
    x = L.Dense(dim)(x)
    x = L.Dropout(dropout_rate)(x)
    return x
#----------------------------------------------------------------
def transformer_encoder(x, num_heads, dim, mlp_dim):
    skip_1 = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(
        num_heads=num_heads, key_dim=dim
    )(x, x)
    x = L.Add()([x, skip_1])

    skip_2 = x
    x = L.LayerNormalization()(x)
    x = mlp(x, mlp_dim, dim)
    x = L.Add()([x, skip_2])

    return x
#---------------------------------------------------------------------------
def vit_block(inputs, dim, patch_size=2, num_layers=1):
    B, H, W, C = inputs.shape

    # 1x1 conv: d-dimension
    x = L.Conv2D(
        filters=dim,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    # Reshape x to flattened patches
    P = patch_size*patch_size
    N = int(H*W//P)
    x = L.Reshape((P, N, dim))(x)

    # Transformr Encoder
    for _ in range(num_layers):
        x = transformer_encoder(x, 2, dim, dim*2)

    # Reshape
    x = L.Reshape((H, W, dim))(x)

    # 1x1 conv: C-dimension
    x = L.Conv2D(
        filters=C,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    return x

#------------------------------------------------------------------------------------------
def RCGNet(input_shape, num_classes):
    num_channels = [16, 32, 64, 64, 64, 96, 144, 128, 192, 160, 240, 640]
    dim = [144, 192, 240]
    num_layers = [2, 4, 3]

    # Input layer
    inputs = L.Input(input_shape)
    # CNN block
    x = base_cnn(inputs)
    # transformer block
    x = vit_block(x, num_channels[6], dim[0], num_layers=num_layers[0])
    x = vit_block(x, num_channels[8], dim[1], num_layers=num_layers[1])

    x = L.Conv2D(
        filters=num_channels[11],
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    x = tf.keras.layers.Dropout(rate=0.2)(x)

    # Classifier
    x = L.GlobalAveragePooling2D()(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs, outputs)
    return model

if __name__== "__main__":
    input_shape = (256, 256,3)
    num_classes = 5

    model = RCGNet(input_shape,num_classes)
    model.summary()