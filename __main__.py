"""
Baseline model -> UNet
"""

import os

from keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dropout,
    concatenate,
    Dense,
)

# from keras.callbacks import ModelCheckpoint
from keras import Model

# import keras.backend as K

log_level = True
print("Logging: ", log_level)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class UNet:
    """
    UNet Class Model
    """

    def __init__(
        self,
        input_height: int = 512,
        input_width: int = 512,
        input_features: int = 9,
        num_inputs: int = 12,
        filter_size: int = 12,
        depth: int = 4,
        output_features: int = 8,
        num_outputs: int = 6,
        logging=log_level,
    ):

        self.input_height = input_height
        self.input_width = input_width
        self.num_inputs = num_inputs
        self.input_features = input_features

        self.input_size = (input_height, input_width, num_inputs * input_features)
        self.filter_size = filter_size
        self.kernel_size = 3
        self.depth = depth

        self.logging = logging

        self.num_outputs = num_outputs
        self.output_features = output_features

        self.output_size = num_outputs * output_features

    def input_layer(self):
        x = Input(self.input_size)
        return x

    def single_conv_2d(self, input_layer, n_filters):
        x = Conv2D(
            filters=n_filters, padding="same", kernel_size=(self.kernel_size, self.kernel_size), activation="sigmoid",
        )(input_layer)
        return x

    def double_conv_2d(self, input_layer, n_filters):
        x = Conv2D(
            filters=n_filters,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="same",
            kernel_initializer="he_normal",
        )(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(
            filters=n_filters,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def deconv_2d(self, input_layer, n_filters, stride=2):
        x = Conv2DTranspose(
            filters=n_filters,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(stride, stride),
            padding="same",
        )(input_layer)
        return x

    def pool_and_drop(self, input_layer, dropout_rate=0.1, pool=2):
        x = MaxPooling2D(pool_size=(pool, pool))(input_layer)
        x = Dropout(rate=dropout_rate)(x)
        return x

    def generate_input_layers(self):
        inputs = []
        for _ in range(self.num_inputs):
            inputs.append(Input((self.input_height, self.input_width, self.input_features)))
        x = concatenate(inputs)
        return inputs, x

    def build_model(self):
        # Initialize the Input
        input_layer, concat_layer = self.generate_input_layers()

        conv2d_layers = []
        pool_layers = []
        for i in range(self.depth):
            if len(conv2d_layers) == 0:
                x = self.double_conv_2d(concat_layer, self.filter_size)
                conv2d_layers.append(x)
            else:
                x = self.double_conv_2d(pool_layers[-1], self.filter_size * (2 ** i))
                conv2d_layers.append(x)

            x = self.pool_and_drop(conv2d_layers[-1])
            pool_layers.append(x)

        mid = self.double_conv_2d(pool_layers[-1], self.filter_size * (2 ** self.depth))

        deconv_layers = []
        for i in range(self.depth - 1, -1, -1):
            if len(deconv_layers) == 0:
                x = self.deconv_2d(mid, self.filter_size * (2 ** i))
                deconv_layers.append(x)
                x = concatenate([conv2d_layers[i], deconv_layers[-1]])
                x = self.double_conv_2d(x, self.filter_size * (2 ** i))
                conv2d_layers.append(x)

            else:
                x = self.deconv_2d(conv2d_layers[-1], self.filter_size * (2 ** i))
                deconv_layers.append(x)
                x = concatenate([conv2d_layers[i], deconv_layers[-1]])
                x = self.double_conv_2d(x, self.filter_size * (2 ** i))

                conv2d_layers.append(x)

        final_layer = self.single_conv_2d(conv2d_layers[-1], self.num_outputs * self.output_features)

        output_layer=[]
        for i in range(self.num_outputs):
            output_layer.append(Dense(self.output_features)(final_layer))
        model = Model(input_layer, output_layer)
        if self.logging:
            print(model.summary())


UNet().build_model()
