import os
import csv
import numpy as np
import cv2

from typing import List, Tuple

def read_voltage(filepath: str, index: List[int]) -> Tuple[List[np.ndarray], List[int]]:
    """
    Read voltage data from a CSV file and return the data for specified indices,
    preserving the order of the indices.

    Args:
        filepath (str): The path to the CSV file containing voltage data.
        index (List[int]): A list of block indices to extract (e.g., [0, 2, 5]).

    Returns:
        Tuple[List[np.ndarray], List[int]]: A list of 2D voltage arrays (float32),
                                            and the indices that were successfully read.
    """
    assert os.path.isfile(filepath), f"Error! {filepath} does not exist or is not a file!"
    print(f"Reading csv file from: {filepath}")

    # Convert list to set for fast lookup
    index_set = set(index)
    block_map = {}
    current_block = []
    block_counter = 0

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for line_idx, line in enumerate(reader):
            if line_idx % 17 == 0:
                # New block header
                if block_counter in index_set and current_block:
                    try:
                        array = np.array(current_block, dtype=np.float32)
                        block_map[block_counter] = array
                    except ValueError:
                        print(f"Warning: Failed to parse block {block_counter}")
                    current_block = []
                block_counter += 1
            else:
                if (block_counter - 1) in index_set:
                    current_block.append(line)

        # Handle last block if unfinished
        if (block_counter - 1) in index_set and current_block:
            try:
                array = np.array(current_block, dtype=np.float32)
                block_map[block_counter - 1] = array
            except ValueError:
                print(f"Warning: Failed to parse block {block_counter - 1}")

    # Reconstruct output in the original index order
    inputdata = []
    final_indices = []
    for idx in index:
        if idx in block_map:
            inputdata.append(block_map[idx])
            final_indices.append(idx)

    return inputdata, final_indices

def load_image(folderpath: str, index: List[int], offset_num: int = 0) -> np.ndarray:
    """
    Load images from a specified directory based on the provided index.
    Args:
        folderpath (str): The directory path where the images are stored.
        index (list): A list of indices corresponding to the images to be loaded.
        offset_num (int): An offset to adjust the image filenames (default is 0).
    Returns:
        np.ndarray: An array of images loaded from the specified directory.
    """
    assert os.path.isdir(folderpath), f"Error! {folderpath} does not exist!"
    images = np.array([cv2.imread(os.path.join(folderpath, f"label{i0 - offset_num}.jpg"), 0) for i0 in index], dtype=int)
    return images