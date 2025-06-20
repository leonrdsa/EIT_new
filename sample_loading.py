"""
This a sample script for loading voltage data (input features) and 
corresponding binarized images (ground truth) from a specified directory structure.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.dataloader import read_voltage, load_image

if __name__ == "__main__":
    'Variables for loading data'
    data_path = './data'
    experiment_name = ['1022.4', '1025.8']

    num_samples = 722   # Total number of samples
    offset_num = 2      # Offset to skip the first two samples
    num_pins = 16       # Number of pins in the voltage data
    resolution = 128    # Image resolution (assumed square, so height = width)
    sampling_rate = 128 # Sampling rate for voltage data

    sample_id = 0
    #------------------------------------------------------------------------------
    'Load voltage data and images'
    img_shape = (0, resolution, resolution)
    voltage_shape = (0, num_pins, sampling_rate*num_pins)
    
    images = np.empty(img_shape, dtype=int)
    voltage_data = np.empty(voltage_shape, dtype=int)
    for folder in experiment_name:
        index = np.arange(offset_num, num_samples+offset_num).tolist()

        exp_path = os.path.join(data_path, folder)
        voltage_path = os.path.join(exp_path, 'voltage.csv')
        img_path = os.path.join(exp_path,'label_vector')

        # Voltage input
        data, index = read_voltage(voltage_path, index)
        voltage_data = np.vstack((voltage_data, data)) 

        # Labelled Vector Images
        images = np.vstack((images,load_image(img_path, index, offset_num)))

    print(f"Voltage data shape: {voltage_data.shape}")
    print(f"Images shape: {images.shape}")

    #------------------------------------------------------------------------------
    'Visualize a sample'

    plt.imshow(images[sample_id, :, :], cmap='gray')
    plt.title('Phantom Binary Image')
    plt.axis('off')
    plt.show()


    plt.plot(voltage_data[sample_id, 1, :], label='Pin 2')
    plt.plot(voltage_data[sample_id, 3, :], label='Pin 4')
    plt.plot(voltage_data[sample_id, 7, :], label='Pin 8')
    plt.plot(voltage_data[sample_id, 11, :], label='Pin 12')
    plt.title('Voltage Data for Pins 2, 4, 8, and 12')
    plt.xlabel('Time Series')
    plt.ylabel('Voltage Raw')
    plt.grid()
    plt.legend()
    plt.show()