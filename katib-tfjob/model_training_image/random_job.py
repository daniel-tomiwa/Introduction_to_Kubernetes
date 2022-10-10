"""An example of multi-worker training with Keras model using Strategy API."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, MaxPool2D , Flatten
from tensorflow.keras.applications import VGG19


def generate_training_data(args):
    """
    function to download the dataset from where it is hosted
    and return the data to be used for training
    """
    os.system(f'wget {args.url}')

    img_size = 224
    data = np.load('/app/data.npy', allow_pickle=True)
    x = []
    y = []
    for feature, label in data:
        x.append(feature)
        y.append(label)
    
    x = np.array(x) / 255
    x = x.reshape(-1, img_size, img_size, 3)
    y = np.array(y)
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)

    data_path = '/train/preprocessed_data'
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    np.save(f'{data_path}/x_train.npy', x_train)  
    np.save(f'{data_path}/x_test.npy', x_test)
    np.save(f'{data_path}/y_train.npy', y_train)
    np.save(f'{data_path}/y_test.npy', y_test)

def get_opt(args):
    if args.optimizer  == "adam":
        opt = tf.keras.optimizers.Adam(
            learning_rate = args.lr
        )
    elif args.optimizer == "sgd":
        opt = tf.keras.optimzer.SGD(
            learning_rate = args.lr
        )
    return opt

def build_and_compile_cnn_model(args):
    #using a pretrained model
    pre_trained_model = VGG19(input_shape=(224,224,3), include_top=False, weights="imagenet")
        #pre_trained_model.trainable = False

    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False

    model = Sequential([
        pre_trained_model,
        MaxPool2D((2,2) , strides = 2),
        Flatten(),
        Dense(5 , activation='softmax')
    ])

    opt = get_opt(args)

    model.compile(
        optimizer = opt, 
        loss = 'categorical_crossentropy', 
        metrics = ['accuracy']
    )

    return model

def main(args):

    # MultiWorkerMirroredStrategy creates copies of all variables in the model's
    # layers on each device across all workers
    # communication_options = tf.distribute.experimental.CommunicationOptions(
    #     implementation = tf.distribute.experimental.CommunicationImplementation.RING
    # )
    # strategy = tf.distribute.MultiWorkerMirroredStrategy(
    #     communication_options = communication_options
    # )

    # BATCH_SIZE_PER_REPLICA = 32
    # BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    BATCH_SIZE = 64

    generate_training_data(args)

    # with strategy.scope():
    #     # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = build_and_compile_cnn_model(args)

    data_path = '/train/preprocessed_data'

    x_train = np.load(f"{data_path}/x_train.npy", allow_pickle=True)
    y_train = np.load(f"{data_path}/y_train.npy", allow_pickle=True)
    x_test =  np.load(f"{data_path}/x_test.npy", allow_pickle=True)
    y_test = np.load(f"{data_path}/y_test.npy", allow_pickle=True)

    # Define the checkpoint directory to store the checkpoints
    # checkpoint_dir = args.checkpoint_dir

    # Name of the checkpoint files
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    # callbacks = [
    #   tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    #   tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
    #                                      save_weights_only=True),
    # ]
    # Keras' `model.fit()` trains the model with specified number of epochs and
    # number of steps per epoch. Note that the numbers here are for demonstration
    # purposes only and may not sufficiently produce a model with good quality.

    multi_worker_model.fit(
        x_train,
        y_train, 
        batch_size = BATCH_SIZE, 
        epochs = 3,
        validation_data = (x_test, y_test)
    )

    # Saving a model
    # Let `is_chief` be a utility function that inspects the cluster spec and
    # current task type and returns True if the worker is the chief and False
    # otherwise.
    # def is_chief():
    #     return TASK_INDEX == 0

    # if is_chief():
    #     model_path = args.saved_model_dir

    # else:
    #     # Save to a path that is unique across workers.
    #     model_path = args.saved_model_dir + '/worker_tmp_' + str(TASK_INDEX)
    model_path = args.saved_model_dir
    multi_worker_model.save(model_path)


if __name__ == '__main__':

    # to decide if a worker is chief, get TASK_INDEX in Cluster info
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    print(tf_config)
    # TASK_INDEX = tf_config['task']['index']

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--saved_model_dir',
        type=str,
        required=True,
        help='Tensorflow export directory.'
    )

    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='Url where the data is hosted'
    )

    # parser.add_argument(
    #     '--checkpoint_dir',
    #     type=str,
    #     required=True,
    #     help='Tensorflow checkpoint directory.'
    # )

    parser.add_argument(
        '--lr',
        type=str,
        required=True,
        help='the learning rate of the model has specified by katib.'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        required=True,
        help='The optimizer to be used for model compilation.'
    )

    parsed_args = parser.parse_args()
    main(parsed_args)