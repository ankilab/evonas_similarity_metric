from genetic_algorithm.src.models_structure.res_layers import RES_2D, BOT_2D
import tensorflow as tf
from tensorflow.keras.layers import  Conv2D, DepthwiseConv2D, Dense, BatchNormalization, GlobalAveragePooling2D, \
    MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, ReLU, Flatten, Dropout, \
    Rescaling
from tensorflow_addons.layers import InstanceNormalization




params={
    "input_shape":(6_000, 1),
    "sample_rate":16_000,
    "nb_classes":12
}

def translate(chromosome: list, params: dict):
    input_tensor = tf.keras.Input(shape=params["input_shape"], name='input_image')  

    for gene in chromosome:
        layer_type = gene["layer"]
        #print(layer_type)
        if layer_type == "C_2D":
            x = Conv2D(gene["filters"], gene["kernel_size"], strides=gene["strides"], padding='same', kernel_initializer="he_normal")(x)
        elif layer_type == "DC_2D":
            x = DepthwiseConv2D( gene["kernel_size"], strides=gene["strides"], padding='same', depthwise_initializer="he_normal")(x)
        elif layer_type =="Rescaling":
            x = Rescaling(gene["scale"])(input_tensor)
        elif layer_type == "BN_2D":
            x = BatchNormalization(epsilon=1.001e-5)(x)
        elif layer_type == "IN_2D":
            x = InstanceNormalization()(x)
        elif layer_type == "R_2D":
            x = ReLU()(x)
        elif layer_type == "MP_2D":
            x= MaxPooling2D(pool_size=gene["pool_size"], padding='same')(x)
        elif layer_type == "AP_2D":
            x= AveragePooling2D(pool_size=gene["pool_size"], padding='same')(x)
        elif layer_type == "RES_2D":
            x = RES_2D(x, gene["filters"], gene["strides"], gene["kernel_size"], gene["skip_connection"])
        elif layer_type == "BOT_2D":
            x = BOT_2D(x, gene["filters"], gene["strides"], gene["kernel_size"], gene["skip_connection"])
        elif layer_type == "GAP_2D":
            x = GlobalAveragePooling2D()(x)
        elif layer_type == "GMP_2D":
            x = GlobalMaxPooling2D()(x)
        elif layer_type == "DO":
            x = Dropout(gene["rate"])(x)
        elif layer_type == "D":
            x = Dense(gene["units"], activation=gene["activation"])(x)
        elif layer_type == "FLAT":
            x = Flatten()(x)

    x = Dense(params["nb_classes"], activation='softmax', name='last_fc')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return model



# Load JSON data from the file
#import json
#with open("Results\\ga_20231011-175234_random_training_speech\\chromosome.json", "r") as json_file:
#    chromosome = json.load(json_file)

# Create the TensorFlow model from JSON data
#model = translate(chromosome, params)

# Print the model summary
#model.summary()
