import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def Voltage2Image(
        input_shape: tf.data.Dataset, 
        output_shape: tf.data.Dataset,
    ) -> tf.keras.Model:
    """
    Build a simple neural network model that maps voltage data to images.
    Args:
        input_shape (tf.data.Dataset): Shape of the input voltage data.
        output_shape (tf.data.Dataset): Shape of the output images.
    Returns:
        tf.keras.Model: A Keras model instance.
    """

    model = Sequential()
    
    model.add(layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size =2))
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Reshape((16,-1)))

    model.add(layers.Bidirectional(layers.LSTM(16, input_shape = input_shape, return_sequences=True)))
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