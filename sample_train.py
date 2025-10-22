import argparse

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from utils.setup import set_seeds
from utils.dataloader import load_eit_dataset
from utils.metrics import compute_segmentation_metrics
from models.image_reconstruction import Voltage2Image
from models.schedulers import scheduler, SchedulerandTrackerCallback

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
    parser.add_argument('-e', '--experiments', nargs='+', 
                        default = ['1022.4'],
                        # default=['1022.1', '1022.2', '1022.3', '1022.4', '1024.5', '1024.6', '1024.7', '1025.8'],
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
    parser.add_argument('--binary-threshold', type=float, default=0.5, help='Threshold to binarize the images')

    # Model training parameters
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--loss', type=str, choices=['mse', 'binary_crossentropy'], default='binary_crossentropy',
                        help='Loss function to use during training')
    parser.add_argument('-l', '--load-model', action='store_true', help='Flag to load a pre-trained model')
    parser.add_argument('--model-path', type=str, default='./checkpoints/modeloriginal.h', help='Path to the pre-trained model file')

    args = parser.parse_args()
    return args

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

    x_train, x_test, y_train, y_test = train_test_split(
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
    # Compiling Model and Optimizer

    callback = SchedulerandTrackerCallback(scheduler)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    if args.load_model:
        model = load_model(args.model_path)
    else:
        model = Voltage2Image(
            input_shape=voltage_data.shape[1:], 
            output_shape=images.shape[1:]
        )
    
        model.build(
            voltage_data.shape
        )

        model.compile(
            loss=args.loss, 
            optimizer=opt, 
            metrics=['accuracy']
        )

        model.summary()

        history = model.fit(
            train_dataset, 
            epochs=args.epochs, 
            validation_data=test_dataset,
            callbacks=callback,
            shuffle=True
        )
        loss = history.history['loss']

        # -------------------------------------------------------------------
        # Heatmap of loss with each epoch and learning rate

        # TODO: Add option to save the model
        # model.save('./checkpoints/modeloriginal.h5')

    # -------------------------------------------------------------------
    # Evaluation
    if not args.load_model:
        plt.figure()
        plt.plot(history.history['loss'], label = 'Training loss')
        plt.plot(history.history['val_loss'], label = 'Validation loss')
        plt.legend()  # Add legend elements
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.show()
    
    metrics = compute_segmentation_metrics(
        model, 
        x_test, 
        y_test, 
        threshold=args.binary_threshold
    )
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.5f}")