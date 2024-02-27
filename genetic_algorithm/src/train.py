import tensorflow as tf
import sys
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import json
import flammkuchen as fl
import matplotlib.pyplot as plt
import ast
import argparse
import time
import warnings
warnings.simplefilter('ignore')

sys.path.insert(0, '.')
sys.path.insert(0, '../.')
sys.path.insert(0, '../../.')

from datasets.get_datasets import get_datasets

#########################################################################################
# Some general configuration
#########################################################################################
# limit GPU memory consumption to enable parallel training of multiple neural networks
# --> Memory limit is not really 1024MB in practice...
#gpus = tf.config.list_physical_devices('GPU')
#tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

# --> This is better, but we never know how many memory exactly is allocated
# However, 24 GB should be enough to train 10 models in parallel, even if we have 10 huge models
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# set weights data type
# policy = tf.keras.mixed_precision.Policy('bfloat16')
# tf.keras.mixed_precision.set_global_policy(policy)

# resolve args
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str)
parser.add_argument("--gen_dir", type=str)
parser.add_argument("--individual_dir", nargs='+')
parser.add_argument("--nb_epochs", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--classes_filter", type=int, nargs="*")

args = parser.parse_args()
#########################################################################################
# Load data
#########################################################################################
#global ds_train, ds_val, ds_test
ds_train, ds_val, ds_test = get_datasets(args.dataset, classes_filter=args.classes_filter)
img, _=next(iter(ds_test))

#########################################################################################
# DNN training
#########################################################################################
# load and compile tf model
def load_tf_model(path):
    m = tf.keras.models.load_model(path, custom_objects={"InstanceNormalization": InstanceNormalization
                                                         }, compile=False)
    return m

def exp_scheduler(epoch, lr):
        if epoch < 2:
            return 0.01
        elif epoch < 4:
            return 0.05
        elif epoch < 6:
            return 0.001
        elif epoch < 8:
            return 0.0005
        elif epoch < 10:
            return 0.0001
        else:
            return lr * np.exp(-0.1)

print("dirs:", args.individual_dir)
for individual in args.individual_dir:
    print("Individual: ",individual )
    model_path = args.results_dir + "/" + args.gen_dir + "/" + individual + "/models/model_untrained.h5"
    
    model = load_tf_model(model_path)
    print("model_loaded")

    ############################################################
    initial_lr = 0.1
    if args.dataset =="imagenet16-120":
        total_samples=151700
    elif args.dataset =="cifar100":
        total_samples=50000
    elif args.dataset =="cifar10":
        total_samples=50000
    steps_per_epoch = total_samples // 128
    print(steps_per_epoch)
    # Create a cosine decay learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_lr, (args.nb_epochs)*steps_per_epoch)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, weight_decay=0.0005)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, weight_decay=0.0005, nesterov=True)
    
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics='accuracy')

    # callback for saving the best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.results_dir + "/" + args.gen_dir + "/" + individual + "/models/model_trained.h5",
        monitor='val_accuracy',
        mode='max',
        save_best_only=True, save_weights_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.05,
        patience=7,
        verbose=0,
        mode="max",
        baseline=None,
        restore_best_weights=False,
    )


    #lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=exp_scheduler, verbose=0)
    callbacks = [ model_checkpoint_callback, early_stopping]#, early_stopping]

    # train
    history = model.fit(ds_train.batch(128),
                        validation_data=ds_val.batch(128),
                        callbacks=callbacks,
                        verbose=0,
                        epochs=args.nb_epochs)
    print("finished")
    #########################################################################################
    # Save training history
    #########################################################################################
    save_path = args.results_dir + "/" + args.gen_dir + "/" + individual + "/history.fl"
    fl.save(save_path, history.history)

    #########################################################################################
    best_val_acc = np.max(history.history['val_accuracy'])
    #########################################################################################
    # Save determined test accuracy in results.json
    #########################################################################################

    start_time = time.time()
    #model.predict(tf.zeros((1, *img.shape.as_list())))
    model.predict(np.expand_dims(img, axis=0))
    end_time = time.time()
    inference_time = end_time - start_time
    print("inference time: ", inference_time)

    #n_parameters=count_params(model.trainable_weights)
    ####################################################
    with open(args.results_dir + "/" + args.gen_dir + "/" + individual + '/results.json') as f:
        d = json.loads(f.read())

    d["val_acc"] = float(best_val_acc)
    d["inference_time"] = float(inference_time)
    #d["n_parameters"] = float(n_parameters)
    with open(args.results_dir + "/" + args.gen_dir + "/" + individual + '/results.json', 'w') as f:
        json.dump(d, f, indent=2)
    
    del model, history, callbacks, model_path, model_checkpoint_callback, best_val_acc, inference_time, d, save_path
