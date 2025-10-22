"""
Sample loader for EIT (Electrical Impedance Tomography) project.

This script demonstrates how to load raw voltage time-series (input features)
and their corresponding binarized labelled images (ground truth) from a
directory layout where each experiment has a `voltage.csv` and a
`label_vector/` folder containing vectorised label images.

The script is intentionally small and focused on showing the expected data
shapes and simple visualization; it is *not* an optimized data pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils.dataloader import read_voltage, load_image

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

    parser.add_argument('--data-path', type=str, default='./data', help='Root data directory')
    parser.add_argument('-e', '--experiments', nargs='+', default=['1022.4', '1025.8'],
                        help='List of experiment folder names to load (space separated)')
    parser.add_argument('--num-samples', type=int, default=722, help='Total number of samples per experiment')
    parser.add_argument('--offset-num', type=int, default=2, help='Offset to skip the first N samples')
    parser.add_argument('--num-pins', type=int, default=16, help='Number of pins in the voltage data')
    parser.add_argument('--resolution', type=int, default=128, help='Image resolution (height == width)')
    parser.add_argument('--sampling-rate', type=int, default=128, help='Sampling rate for voltage data')
    parser.add_argument('--sample-id', type=int, default=0, help='Index of sample to visualize (0-based)')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = read_options()

    # ------------------------------------------------------------------
    # Prepare empty containers for stacking loaded data from multiple
    # experiments. We start with zero-sized arrays and vertically stack
    # each experiment's data using `np.vstack` below.
    # ------------------------------------------------------------------
    # Image shape: (N, H, W) where N will grow as we append experiments
    img_shape = (0, args.resolution, args.resolution)
    # Voltage shape: (N, pins, samples_per_pin)
    voltage_shape = (0, args.num_pins, args.sampling_rate * args.num_pins)

    images = np.empty(img_shape, dtype=int)
    voltage_data = np.empty(voltage_shape, dtype=int)

    # Loop over requested experiment folders and load data
    for folder in args.experiments:
        # `index` is a list of sample indices to load for this experiment.
        # `offset_num` is used because some datasets may reserve the first
        # N rows for metadata or a different numbering scheme.
        index = np.arange(args.offset_num, args.num_samples + args.offset_num).tolist()

        exp_path = os.path.join(args.data_path, folder)
        voltage_path = os.path.join(exp_path, 'voltage.csv')
        img_path = os.path.join(exp_path, 'label_vector')

        # Read voltage time-series for the requested indices. `read_voltage`
        # is expected to return (data_array, updated_index) where `data_array`
        # has shape (n_samples, n_pins, n_timepoints).
        data, index = read_voltage(voltage_path, index)
        voltage_data = np.vstack((voltage_data, data))

        # Read the binarized label vectors (images) for the same indices.
        # `load_image` should return an array shaped (n_samples, H, W).
        images = np.vstack((images, load_image(img_path, index, args.offset_num)))

    # Print shapes so the user can verify successful loading
    print(f"Voltage data shape: {voltage_data.shape}")
    print(f"Images shape: {images.shape}")

    # ------------------------------------------------------------------
    # Quick visualization of a single sample. This is intended for sanity
    # checking: show the binary label image and a few voltage traces.
    # ------------------------------------------------------------------
    # Show the binary image for the chosen sample index
    plt.imshow(images[args.sample_id, :, :], cmap='gray')
    plt.title('Phantom Binary Image')
    plt.axis('off')
    plt.show()

    # Plot example voltage traces. Note: here indices are 0-based so
    # `voltage_data[..., 1, :]` corresponds to "Pin 2" in the original
    # script's labeling convention.
    plt.plot(voltage_data[args.sample_id, 1, :], label='Pin 2')
    plt.plot(voltage_data[args.sample_id, 3, :], label='Pin 4')
    plt.plot(voltage_data[args.sample_id, 7, :], label='Pin 8')
    plt.plot(voltage_data[args.sample_id, 11, :], label='Pin 12')
    plt.title('Voltage Data for Pins 2, 4, 8, and 12')
    plt.xlabel('Time Series')
    plt.ylabel('Voltage Raw')
    plt.grid()
    plt.legend()
    plt.show()