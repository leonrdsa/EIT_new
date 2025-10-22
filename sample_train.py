import argparse

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback

from utils.setup import set_seeds
from utils.dataloader import load_eit_dataset

def read_options() -> argparse.Namespace:
    """Parse and return command-line options.

    Returns:
        argparse.Namespace: parsed arguments with attributes matching the
        original hard-coded variables (data_path, experiments, num_samples, ...)
    """

    # Create the parser and register arguments. Defaults preserve original
    # behavior so existing workflows won't be broken.
    parser = argparse.ArgumentParser(
        description='Load EIT voltage data and corresponding labelled images'
    )

    # Data loading parameters
    parser.add_argument('--data-path', type=str, default='./data', help='Root data directory')
    parser.add_argument('-e', '--experiments', nargs='+', default=['1022.4', '1025.8'],
                        help='List of experiment folder names to load (space separated)')
    parser.add_argument('--num-samples', type=int, default=722, help='Total number of samples per experiment')
    parser.add_argument('--offset-num', type=int, default=2, help='Offset to skip the first N samples')
    parser.add_argument('--num-pins', type=int, default=16, help='Number of pins in the voltage data')
    parser.add_argument('--resolution', type=int, default=128, help='Image resolution (height == width)')
    parser.add_argument('--sampling-rate', type=int, default=128, help='Sampling rate for voltage data')
    parser.add_argument('--sample-id', type=int, default=0, help='Index of sample to visualize (0-based)')

    # Data processing parameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training/testing datasets')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting.')

    # Model training parameters
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')

    args = parser.parse_args()
    return args

relu = lambda x: tf.math.maximum(x,0.0)

def scheduler(epoch, lr):
    if epoch%500 == 0:
        return lr*tf.math.exp(-0.1)
    else:
        return lr

class SchedulerandTrackerCallback(Callback):
    def __init__(self,scheduler):
        self.scheduler = scheduler
        self.epoch_lr = []
        self.epoch_loss = []
        
    def on_epoch_begin(self,epoch,logs = None):
        current_lr = self.model.optimizer.learning_rate.numpy()
        new_lr = self.scheduler(epoch,current_lr)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
    def on_epoch_end(self,epoch,logs = None):
        current_lr = self.model.optimizer.learning_rate.numpy()
        loss = logs.get('loss')
        self.epoch_lr.append(current_lr)
        self.epoch_loss.append(loss)

def model2(input_shape, output_shape):
    model = Sequential()
    
    model.add(layers.Conv2D(16,kernel_size=3, activation = relu, input_shape = input_shape,))
    model.add(layers.MaxPooling2D(pool_size =2))
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Reshape((16,-1)))
            
    model.add(layers.Bidirectional(layers.LSTM(16,input_shape = input_shape, return_sequences=True)))
    model.add(layers.Dense(64, activation = 'sigmoid'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(256, activation = 'sigmoid'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1024, activation = 'sigmoid'))
    model.add(layers.Dropout(0.25))

    
    model.add(layers.Flatten())
    model.add(layers.Dense(output_shape[0]*output_shape[1], activation = 'sigmoid'))
    model.add(layers.Reshape(output_shape))
    return model    

if __name__ == "__main__":

    args = read_options()

    set_seeds(args.seed)

    #------------------------------------------------------------------
    # Loading Samples

    voltage_data, images = load_eit_dataset(
        data_path=args.data_path,
        experiments=args.experiments,
        num_samples=args.num_samples,
        offset_num=args.offset_num,
        num_pins=args.num_pins,
        resolution=args.resolution,
        sampling_rate=args.sampling_rate
    )

    voltage_data = voltage_data[..., np.newaxis]  # Add channel dimension for TF

    # Print shapes so the user can verify successful loading
    print(f"Voltage data shape: {voltage_data.shape}")
    print(f"Images shape: {images.shape}")

    # ------------------------------------------------------------------
    # Processing of Images

    images = 1*(images > 100) # Binarize images

    # ------------------------------------------------------------------
    # Splits

    x_train ,x_test, y_train, y_test = train_test_split(
        voltage_data, 
        images, 
        test_size=args.test_size, 
        random_state=args.seed # not included by set_seeds function
    ) # split the dataset

    # normalization values (along sample and time axis)
    mean = x_train.mean(axis=(0,2), keepdims = True)
    var = x_train.std(axis=(0,2), keepdims = True)

    x_train = (x_train - mean) / var
    x_test = (x_test - mean) / var

    # convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)

    print(f"Train dataset batches: {len(train_dataset)}, Test dataset batches: {len(test_dataset)}")
    
    # -------------------------------------------------------------------
    # Compiling Model

    'Setting learningrate scheduler and record loss and its learning rate'
    # callback = LearningRateScheduler(scheduler)
    callback = SchedulerandTrackerCallback(scheduler)
    model = model2(input_shape=voltage_data.shape[1:], output_shape=images.shape[1:])
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.build(voltage_data.shape)
    # '------------------------------------------------------'
    'Try out differrent loses'
    # model.compile(loss = "mse", optimizer = opt)
    model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy']) #If it is binary maybe this will work better
    model.summary()

    history = model.fit(train_dataset, epochs=args.epochs, validation_data= test_dataset,callbacks = callback,shuffle = True)
    loss = history.history['loss']

    # -------------------------------------------------------------------
    'Heatmap of loss with each epoch and learning rate'

    plt.figure()
    plt.plot(history.history['loss'], label = 'Training loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.legend()  # Add legend elements
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')