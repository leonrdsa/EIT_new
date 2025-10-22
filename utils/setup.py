from typing import Optional, Union

def set_seeds(seed: Union[int, None]) -> None:
    """
    Set random seeds for reproducible experiments across multiple libraries.
    
    Sets seeds for Python's random module, NumPy, and TensorFlow to ensure
    deterministic behavior in machine learning experiments. This is crucial
    for reproducible research and debugging.
    
    Args:
        seed: Random seed value. If None, no seeds are set.
        
    Note:
        This function imports the required libraries locally to avoid
        unnecessary dependencies if not needed. Also configures TensorFlow
        for deterministic operations and handles transformers library if available.
    """
    if seed is None:
        return
        
    import numpy as np
    import random
    import tensorflow as tf
    import os

    # Set seeds for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'