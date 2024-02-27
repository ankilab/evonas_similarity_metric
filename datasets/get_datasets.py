import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from scipy import signal
import itertools
import os
import pandas as pd
import re
from pathlib import Path


def get_datasets(dataset, samples=6_000, classes_filter=None):
    def _filter_classes(image, label):
        return tf.reduce_any(tf.equal(label, classes_filter))

    def _predicate(x, label, allowed_labels=classes_filter):
        allowed_labels = tf.constant(allowed_labels)
        isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
        reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
        return tf.greater(reduced, tf.constant(0.))

    if dataset == "speech_commands":
        """
        [0: 'down', 1: 'go', 2: 'left', 3: 'no', 4: 'off', 5: 'on', 6: 'right', 7: 'stop', 8: 'up', 9: 'yes', 
        10: '_silence_', 11: '_unknown_']
        """
        #ds_train = tfds.load("speech_commands", data_dir='datasets/', split='train', as_supervised=True, download=False)
        #ds_val = tfds.load("speech_commands", data_dir='datasets/', split='validation', as_supervised=True, download=False)
        #ds_test = tfds.load("speech_commands", data_dir='datasets/', split='test', as_supervised=True, download=False)
        (ds_train,ds_val, ds_test) = tfds.load(
            "speech_commands",
            data_dir='datasets/',
            split=['train','validation', 'test'],
            as_supervised=True,
            download=False,
            with_info=False
        )
        num_examples = ds_train.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        num_examples_val = ds_val.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        num_examples_test = ds_test.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        print(f"Number of training examples: {num_examples}")
        print(f"Number of validation examples: {num_examples_val}")
        print(f"Number of test examples: {num_examples_test}")

        nb_classes = 12
        def _prepare_dataset(ds, normalize=True):
                if len(classes_filter) != 0:
                    ds = ds.filter(_predicate)
                    ds = ds.map(lambda x, y: (tf.py_function(_resample_func, [x, samples], Tout=tf.float32), tf.one_hot(tf.where(tf.equal(y, classes_filter))[0], len(classes_filter))[0]))
                else:
                    ds = ds.map(lambda x, y: (tf.py_function(_resample_func, [x, samples], Tout=tf.float32), tf.one_hot(y, nb_classes)))
                if normalize:
                    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
                return ds


        ds_train = _prepare_dataset(ds_train)
        ds_val = _prepare_dataset(ds_val)
        ds_test = _prepare_dataset(ds_test)

        num_examples = ds_train.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        num_examples_val = ds_val.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        num_examples_test = ds_test.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        print(f"Number of training examples: {num_examples}")
        print(f"Number of validation examples: {num_examples_val}")
        print(f"Number of test examples: {num_examples_test}")

        return ds_train, ds_val, ds_test
    elif dataset == "motion_sense":
        path = Path("datasets/motion_sense_accelerometer")
        classes = {'dws': 0, 'ups': 1, 'sit': 2, 'std': 3, 'wlk': 4, 'jog': 5}
        train_idxs, val_idxs, test_idxs = np.arange(0, 19), np.arange(19, 22), np.arange(22, 25)

        def _get_ds(data, labels, samples=4_000):
            ds = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(data, labels),
                                                output_types=(tf.float64, tf.int32),
                                                output_shapes=((samples, 1), ()))
            if classes_filter is not None:
                ds = ds.map(lambda x, y: (x, tf.one_hot(tf.where(tf.equal(y, classes_filter))[0], len(classes_filter))[0]))
            else:
                ds = ds.map(lambda x, y: (x, tf.one_hot(y, len(classes))))
            return ds

        train_data, val_data, test_data = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        for folder in os.listdir(path):
            class_path = path / folder
            for f in os.listdir(class_path):
                df = pd.read_csv(class_path / f)

                x, y, z = np.asarray(df['x']), np.asarray(df['y']), np.asarray(df['z'])
                magnitude = list(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
                magnitude = _resample_func(magnitude, samples)
                label = classes[folder[0:3]]

                idx = int(re.findall(r'\d+', f)[0])
                if idx in train_idxs:
                    train_data.append(magnitude)
                    train_labels.append(label)
                elif idx in val_idxs:
                    val_data.append(magnitude)
                    val_labels.append(label)
                else:
                    test_data.append(magnitude)
                    test_labels.append(label)

        ds_train = _get_ds(np.asarray(train_data), np.asarray(train_labels), samples)
        ds_val = _get_ds(np.asarray(val_data), np.asarray(val_labels), samples)
        ds_test = _get_ds(np.asarray(test_data), np.asarray(test_labels), samples)

        return ds_train, ds_val, ds_test
    elif dataset=="cifar10":
        (ds_train, ds_test) = tfds.load(
            "cifar10",
            data_dir='datasets/',
            split=['train', 'test'],
            as_supervised=True,
            download=True,
            with_info=False
        )
        if len(classes_filter) != 0:
            # Filter the classes in the training set
            ds_train = ds_train.filter(_filter_classes)
            # Filter the classes in the test set
            ds_test = ds_test.filter(_filter_classes)
            num_classes = len(classes_filter)
        else:
            num_classes = 10

        ds_train = ds_train.map(lambda image, label: (image, tf.one_hot(label, num_classes)))
        ds_test = ds_test.map(lambda image, label: (image, tf.one_hot(label, num_classes)))

        # Calculate the number of examples for training and validation after filtering
        #num_examples = ds_train.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        #num_train_examples = int(0.8 * num_examples)  # 80% for training
        #num_val_examples = num_examples - num_train_examples  # 20% for validation

        # Split the 'train' dataset into training and validation sets
        # ds_train = ds_train.shuffle(buffer_size=num_examples, reshuffle_each_iteration=True)
        #ds_val = ds_train.skip(num_train_examples).take(num_val_examples)
        #ds_train = ds_train.take(num_train_examples)

        def augment_data(image, label):
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Random vertical flip
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_crop(image, size=[32, 32, 3])
            # Padding
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            # Random crop after padding
            image = tf.image.random_crop(image, size=[32, 32, 3])       
            return image, label

        ds_train=ds_train.map(augment_data)
        # Display the number of examples in each set
        #print(f"Number of training examples: {num_train_examples}")
        #print(f"Number of validation examples: {num_val_examples}")
        #print(f"Number of test examples: {ds_test.reduce(np.int64(0), lambda x, _: x + 1).numpy()}")
        ds_val=ds_test
        return ds_train, ds_val, ds_test
    elif dataset=="cifar100":
        (ds_train, ds_test) = tfds.load(
            "cifar100",
            data_dir='datasets/',
            split=['train', 'test'],
            as_supervised=True,
            download=False,
            with_info=False
        )
        if len(classes_filter) != 0:
            # Filter the classes in the training set
            ds_train = ds_train.filter(_filter_classes)
            # Filter the classes in the test set
            ds_test = ds_test.filter(_filter_classes)
            num_classes = len(classes_filter)
        else:
            num_classes = 100
        # Convert labels to one-hot encoded format
        ds_train = ds_train.map(lambda image, label: (image, tf.one_hot(label, num_classes)))
        ds_test = ds_test.map(lambda image, label: (image, tf.one_hot(label, num_classes)))

        # Calculate the number of examples for training and validation after filtering
        #num_examples = ds_train.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        #num_train_examples = int(0.8 * num_examples)  # 80% for training
        #num_val_examples = num_examples - num_train_examples  # 20% for validation

        # Split the 'train' dataset into training and validation sets
        # ds_train = ds_train.shuffle(buffer_size=num_examples, reshuffle_each_iteration=True)
        #ds_val = ds_train.skip(num_train_examples).take(num_val_examples)
        #ds_train = ds_train.take(num_train_examples)

        def augment_data(image, label):
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Random vertical flip
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_crop(image, size=[32, 32, 3])
            # Padding
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            # Random crop after padding
            image = tf.image.random_crop(image, size=[32, 32, 3])       
            return image, label

        ds_train=ds_train.map(augment_data)
        # Display the number of examples in each set
        #print(f"Number of training examples: {num_train_examples}")
        #print(f"Number of validation examples: {num_val_examples}")
        #print(f"Number of test examples: {ds_test.reduce(np.int64(0), lambda x, _: x + 1).numpy()}")
        ds_val=ds_test
        return ds_train, ds_val, ds_test

    elif dataset=="imagenet16-120":
        (ds_train, ds_val) = tfds.load(
            "imagenet_resized/16x16",
            data_dir='datasets/',
            split=['train', 'validation'],
            as_supervised=True,
            download=False,
            with_info=False
        )
        if len(classes_filter) != 0:
            # Filter the classes in the training set
            ds_train = ds_train.filter(_filter_classes)
            # Filter the classes in the test set
            ds_val = ds_val.filter(_filter_classes)
            num_classes = len(classes_filter)
        else:
            num_classes = 1000
        # Convert labels to one-hot encoded format
        ds_train = ds_train.map(lambda image, label: (image, tf.one_hot(label, num_classes)))
        ds_val = ds_val.map(lambda image, label: (image, tf.one_hot(label, num_classes)))

        num_examples = 151700
        num_val_examples = 3000
        num_test_examples = 3000

        def augment_data(image, label):
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Random vertical flip
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_crop(image, size=[16, 16, 3])
            # Padding
            image = tf.image.resize_with_crop_or_pad(image, 20, 20)
            # Random crop after padding
            image = tf.image.random_crop(image, size=[16, 16, 3])       
            return image, label

        ds_train=ds_train.map(augment_data)
        # Split the 'train' dataset into training and validation sets
        #ds_val = ds_val.shuffle(buffer_size=6000, reshuffle_each_iteration=True)
        #ds_test = ds_val.skip(num_val_examples).take(num_test_examples)
        #ds_val = ds_val.take(num_val_examples)
        ds_test=ds_val
        return ds_train, ds_val, ds_test
    
    elif dataset=="fashion_mnist":
        (ds_train, ds_test) = tfds.load(
            "fashion_mnist",
            data_dir='datasets/',
            split=['train', 'test'],
            as_supervised=True,
            download=False,
            with_info=False
        )
        if len(classes_filter) != 0:
            # Filter the classes in the training set
            ds_train = ds_train.filter(_filter_classes)
            # Filter the classes in the test set
            ds_test = ds_test.filter(_filter_classes)
            num_classes = len(classes_filter)
        else:
            num_classes = 10
        # Convert labels to one-hot encoded format
        ds_train = ds_train.map(lambda image, label: (image, tf.one_hot(label, num_classes)))
        ds_test = ds_test.map(lambda image, label: (image, tf.one_hot(label, num_classes)))

        # Calculate the number of examples for training and validation after filtering
        num_examples = ds_train.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        num_train_examples = int(0.8 * num_examples)  # 80% for training
        num_val_examples = num_examples - num_train_examples  # 20% for validation

        # Split the 'train' dataset into training and validation sets
        ds_train = ds_train.shuffle(buffer_size=num_examples, reshuffle_each_iteration=True)
        ds_val = ds_train.skip(num_train_examples).take(num_val_examples)
        ds_train = ds_train.take(num_train_examples)
        
        # Display the number of examples in each set
        print(f"Number of training examples: {num_train_examples}")
        print(f"Number of validation examples: {num_val_examples}")
        print(f"Number of test examples: {ds_test.reduce(np.int64(0), lambda x, _: x + 1).numpy()}")

        return ds_train, ds_val, ds_test
    elif dataset=="mnist":
        (ds_train, ds_test) = tfds.load(
            "mnist",
            data_dir='datasets/',
            split=['train', 'test'],
            as_supervised=True,
            download=True,
            with_info=False
        )
        if len(classes_filter) != 0:
            # Filter the classes in the training set
            ds_train = ds_train.filter(_filter_classes)
            # Filter the classes in the test set
            ds_test = ds_test.filter(_filter_classes)
            num_classes = len(classes_filter)
        else:
            num_classes = 10
        # Convert labels to one-hot encoded format
        ds_train = ds_train.map(lambda image, label: (image, tf.one_hot(label, num_classes)))
        ds_test = ds_test.map(lambda image, label: (image, tf.one_hot(label, num_classes)))

        # Calculate the number of examples for training and validation after filtering
        num_examples = ds_train.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        num_train_examples = int(0.8 * num_examples)  # 80% for training
        num_val_examples = num_examples - num_train_examples  # 20% for validation

        # Split the 'train' dataset into training and validation sets
        ds_train = ds_train.shuffle(buffer_size=num_examples, reshuffle_each_iteration=True)
        ds_val = ds_train.skip(num_train_examples).take(num_val_examples)
        ds_train = ds_train.take(num_train_examples)
        
        # Display the number of examples in each set
        print(f"Number of training examples: {num_train_examples}")
        print(f"Number of validation examples: {num_val_examples}")
        print(f"Number of test examples: {ds_test.reduce(np.int64(0), lambda x, _: x + 1).numpy()}")

        return ds_train, ds_val, ds_test
    elif dataset=="caltech101":
        (ds_test, ds_train) = tfds.load(
            "caltech101",
            data_dir='datasets/',
            split=['train', 'test'],
            as_supervised=True,
            download=False,
            with_info=False
        )
        if len(classes_filter) != 0:
            # Filter the classes in the training set
            ds_train = ds_train.filter(_filter_classes)
            # Filter the classes in the test set
            ds_test = ds_test.filter(_filter_classes)
            num_classes = len(classes_filter)
        else:
            num_classes = 102
        # Convert labels to one-hot encoded format
        ds_train = ds_train.map(lambda image, label: (image, tf.one_hot(label, num_classes)))
        ds_test = ds_test.map(lambda image, label: (image, tf.one_hot(label, num_classes)))

        # Calculate the number of examples for training and validation after filtering
        def preprocess_image(image, label, target_size=(250, 250)):
            # Resize the image to the target size
            #image = tf.image.resize(images=image, size=target_size, preserve_aspect_ratio=True)
            image = tf.image.resize(images=image, size=target_size)

            # Normalize pixel values to [0, 1]
            #image = tf.cast(image, tf.float32) / 255.0
            return image, label
        target_size = (250, 250)   
        ds_train = ds_train.map(lambda image, label: preprocess_image(image, label, target_size))
        ds_test = ds_test.map(lambda image, label: preprocess_image(image, label, target_size))

        num_examples = ds_test.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        num_train_examples = int(0.6 * num_examples)  # 80% for training
        num_val_examples = num_examples - num_train_examples  # 20% for validation

        # Split the 'train' dataset into training and validation sets
        ds_test = ds_test.shuffle(buffer_size=num_examples, reshuffle_each_iteration=True)
        ds_val = ds_test.skip(num_train_examples).take(num_val_examples)
        ds_test = ds_test.take(num_train_examples)
        
        # Display the number of examples in each set
        print(f"Number of training examples: {num_train_examples}")
        print(f"Number of validation examples: {num_val_examples}")
        print(f"Number of test examples: {ds_test.reduce(np.int64(0), lambda x, _: x + 1).numpy()}")

        return ds_train, ds_val, ds_test

    else:
        raise ValueError(f"Given dataset ({dataset}) is not available.")
def _resample_func(x, samples):
    x = signal.resample(x, samples, axis=0)
    return x[..., np.newaxis]


def _normalize(data, label):
    data /= tf.math.reduce_max(tf.abs(data), axis=0)
    return data, label

#ds_train,ds_val,ds_test=get_datasets("imagenet16-120", classes_filter=list(range(120)))
#print(t)
#print(v)
#print(test)
#num_examples = ds_train.reduce(np.int64(0), lambda x, _: x + 1).numpy()
#num_examples_val = ds_val.reduce(np.int64(0), lambda x, _: x + 1).numpy()
#num_examples_test = ds_test.reduce(np.int64(0), lambda x, _: x + 1).numpy()
#print(f"Number of training examples: {num_examples}")
#print(f"Number of validation examples: {num_examples_val}")
#print(f"Number of test examples: {num_examples_test}")