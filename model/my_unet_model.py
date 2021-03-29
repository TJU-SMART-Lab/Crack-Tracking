from keras.models import Model
from keras.layers import *
from keras import backend as K

import tensorflow as tf


def my_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    down_conv_0 = inception_block(inputs, 32)
    down_attention_0a = cbam_attention(down_conv_0, dropout_ratio=0.5)
    down_attention_0b = cbam_attention(down_attention_0a)
    down_pool_0 = MaxPooling2D(pool_size=(2, 2))(down_attention_0b)

    down_conv_1 = inception_block(down_pool_0, 64)
    down_attention_1a = cbam_attention(down_conv_1, dropout_ratio=0.5)
    down_attention_1b = cbam_attention(down_attention_1a)
    down_pool_1 = MaxPooling2D(pool_size=(2, 2))(down_attention_1b)

    down_conv_2 = inception_block(down_pool_1, 128)
    down_attention_2a = cbam_attention(down_conv_2, dropout_ratio=0.5)
    down_attention_2b = cbam_attention(down_attention_2a)
    down_pool_2 = MaxPooling2D(pool_size=(2, 2))(down_attention_2b)

    down_conv_3 = inception_block(down_pool_2, 256)
    down_attention_3a = cbam_attention(down_conv_3, dropout_ratio=0.5)
    down_attention_3b = cbam_attention(down_attention_3a)
    down_pool_3 = MaxPooling2D(pool_size=(2, 2))(down_attention_3b)

    down_conv_4 = inception_block(down_pool_3, 512)
    down_attention_4a = cbam_attention(down_conv_4, dropout_ratio=0.5)
    down_attention_4b = cbam_attention(down_attention_4a)

    up_upsampling_3 = transpose_conv_block(down_attention_4b, 1024, 3)
    up_conv_3 = inception_block(up_upsampling_3, 256)
    up_attention_3a = up_conv_3
    # up_attention_3a = cbam_attention(up_conv_3, dropout_ratio=0.5)
    # up_attention_3b = up_attention_3a
    up_attention_3b = cbam_attention(up_attention_3a)
    up_merge_3 = Concatenate()([down_conv_3, up_attention_3b])

    up_upsampling_2 = transpose_conv_block(up_merge_3, 512, 3)
    up_conv_2 = inception_block(up_upsampling_2, 128)
    up_attention_2a = up_conv_2
    # up_attention_2a = cbam_attention(up_conv_2, dropout_ratio=0.5)
    # up_attention_2b = up_attention_2a
    up_attention_2b = cbam_attention(up_attention_2a)
    up_merge_2 = Concatenate()([down_conv_2, up_attention_2b])

    up_upsampling_1 = transpose_conv_block(up_merge_2, 256, 3)
    up_conv_1 = inception_block(up_upsampling_1, 64)
    up_attention_1a = up_conv_1
    # up_attention_1a = cbam_attention(up_conv_1, dropout_ratio=0.5)
    # up_attention_1b = up_attention_1a
    up_attention_1b = cbam_attention(up_attention_1a)
    up_merge_1 = Concatenate()([down_conv_1, up_attention_1b])

    up_upsampling_0 = transpose_conv_block(up_merge_1, 128, 3)
    up_conv_0 = inception_block(up_upsampling_0, 32)
    up_attention_0a = up_conv_0
    # up_attention_0a = cbam_attention(up_conv_0, dropout_ratio=0.5)
    # up_attention_0b = up_attention_0a
    up_attention_0b = cbam_attention(up_attention_0a)
    up_merge_0 = Concatenate()([down_conv_0, up_attention_0b])

    outputs = Conv2D(2, 1, activation="sigmoid", kernel_initializer="he_normal")(
        up_merge_0
    )

    model = Model(inputs=inputs, outputs=outputs)
    return model


def inception_block(x, filters):
    conv_0 = conv_block(x, filters, 1)

    conv_1 = conv_block(x, filters, 3)

    conv_2_0 = conv_block(x, filters, 3)
    conv_2 = conv_block(conv_2_0, filters, 3)

    conv_3_0 = conv_block(x, filters, 3)
    conv_3_1 = conv_block(conv_3_0, filters, 3)
    conv_3 = conv_block(conv_3_1, filters, 3)

    concat = Concatenate()([conv_0, conv_1, conv_2, conv_3])

    return concat


def cbam_attention(x, ratio=8, dropout_ratio: float = 0):
    cbam_feature = cbam_channel_attention(x, ratio, dropout_ratio)
    cbam_feature = cbam_spatial_attention(cbam_feature)

    return cbam_feature


def cbam_channel_attention(x, ratio=8, dropout_ratio: float = 0):
    channel = x.shape[-1]
    shared_layer_one = Dense(
        channel // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_layer_two = Dense(
        channel, kernel_initializer="he_normal", use_bias=True, bias_initializer="zeros"
    )

    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    # avg_pool = Dropout(0.5)(avg_pool)

    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    # max_pool = Dropout(0.5)(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation("sigmoid")(cbam_feature)

    if dropout_ratio == 0:
        return Multiply()([x, cbam_feature])
        # return x
    else:
        assert 0 < dropout_ratio < 1, "dropout_ratio must be between 0 and 1"
        return Lambda(feature_dropout)([x, cbam_feature, dropout_ratio])
        # return x


def cbam_spatial_attention(x):
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=7,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return Multiply()([x, cbam_feature])


def feature_dropout(x):
    original_feature_map = x[0]
    cbam_feature = x[1]
    dropout_ratio = x[2]

    rank = tf.nn.top_k(
        cbam_feature, k=int(cbam_feature.shape[-1] * dropout_ratio)
    ).indices[0, 0, 0]

    return tf.gather(original_feature_map, rank, axis=-1)


def conv_block(x, filters, kernel_size, activation="relu", padding="same"):
    conv = Conv2D(
        filters,
        kernel_size,
        padding=padding,
        kernel_initializer="he_normal",
        activation=activation,
    )(x)
    # gn = tfa.layers.GroupNormalization()(conv)

    return conv


def transpose_conv_block(x, filters, kernel_size, activation="relu", padding="same"):
    conv = Conv2DTranspose(
        filters,
        kernel_size,
        strides=2,
        padding=padding,
        kernel_initializer="he_normal",
        activation=activation,
    )(x)

    return conv


if __name__ == "__main__":
    model = my_unet()
    # model.summary()
    # plot_model(model, to_file="12345.png", show_shapes=True)
