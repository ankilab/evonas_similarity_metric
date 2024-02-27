from tensorflow.keras.layers import Add, Conv2D, BatchNormalization, ReLU


def RES_2D(x, filters, strides=1, kernel_size=3, skip_connection=True):
    """
    Helper function to build a residual block with two convolutional layers.
    """
    if skip_connection:
        shortcut = x

    # First convolutional layer
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # If the number of filters in the shortcut is different, perform 1x1 convolution to match shapes
    if skip_connection:
        if shortcut.shape[-1] != filters or strides != 1:
            shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        # Add the shortcut to the output
        x = Add()([x, shortcut])
    x = ReLU()(x)

    return x

def BOT_2D(x, filters, strides=1,kernel_size=3, skip_connection=True):
    """
    Helper function to build a residual block with two convolutional layers.
    """
    if skip_connection:
        shortcut = x

    # First convolutional layer
    x = Conv2D(filters, 1, strides=strides, padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = ReLU()(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Third convolutional layer
    x = Conv2D(4*filters, 1, padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)

    if skip_connection:
        # If the number of filters in the shortcut is different, perform 1x1 convolution to match shapes
        if shortcut.shape[-1] != 4*filters or strides != 1:
            shortcut = Conv2D(4*filters, 1, strides=strides)(shortcut)
            shortcut = BatchNormalization( epsilon=1.001e-5)(shortcut)

        # Add the shortcut to the output
        x = Add()([x, shortcut])
    x = ReLU()(x)
    return x