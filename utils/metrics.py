import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from typing import Dict, Union

def compute_segmentation_metrics(
        model: tf.keras.Model, 
        input_data: np.ndarray, 
        image_labels: np.ndarray, 
        threshold=0.5
    ) -> Dict[str, float]:
    """
    Compute segmentation metrics for a trained model on given input data
    Args:
        model (tf.keras.Model): Trained TensorFlow Keras model to evaluate.
        input_data (np.ndarray): Input voltage data for evaluation.
        image_labels (np.ndarray): Ground truth labelled images.
        threshold (float): Threshold to binarize the model's output.
    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """

    res = model.predict(input_data)
    binary_reconstruction = np.where(res >= threshold, 1, 0)
    image_labels_flat = image_labels.reshape(-1)
    binary_reconstruction_flat = binary_reconstruction.reshape(-1)
    accu = np.mean(np.equal(image_labels_flat,binary_reconstruction_flat))
    accu1 = precision_score(image_labels_flat,binary_reconstruction_flat)
    accu2 = recall_score(image_labels_flat,binary_reconstruction_flat)
    f1 = f1_score(image_labels_flat,binary_reconstruction_flat)

    metrics = {
        "accuracy": accu,
        "precision": accu1,
        "recall": accu2,
        "f1-score": f1
    }

    return metrics

def compute_confusion_matrix(
        model: tf.keras.Model, 
        input_data: np.ndarray, 
        image_labels: np.ndarray, 
        threshold=0.5
    ) -> np.ndarray:
    """
    Compute confusion matrix for a trained model on given input data
    Args:
        model (tf.keras.Model): Trained TensorFlow Keras model to evaluate.
        input_data (np.ndarray): Input voltage data for evaluation.
        image_labels (np.ndarray): Ground truth labelled images.
        threshold (float): Threshold to binarize the model's output.
    Returns:
        np.ndarray: Confusion matrix as a 2D numpy array.
    """

    res = model.predict(input_data)
    binary_reconstruction = np.where(res >= threshold, 1, 0)
    image_labels_flat = image_labels.reshape(-1)
    binary_reconstruction_flat = binary_reconstruction.reshape(-1)

    cm = confusion_matrix(image_labels_flat, binary_reconstruction_flat)

    return cm

def compute_MSE(
        model: tf.keras.Model, 
        input_data: np.ndarray, 
        image_labels: np.ndarray
    ) -> float:
    """
    Compute Mean Squared Error (MSE) between two images.
    Args:
        model (tf.keras.Model): Trained TensorFlow Keras model to evaluate.
        input_data (np.ndarray): Input voltage data for evaluation.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        float: MSE value between the two images.
    """
    res = model.predict(input_data)
    mse_value = np.mean((image_labels -  res) ** 2)
    return mse_value

def compute_CNR(
        model: tf.keras.Model, 
        input_data: np.ndarray, 
        image_labels: np.ndarray,
        threshold=0.5
    ) -> float:
    """
    Compute Contrast-to-Noise Ratio (CNR) between two images.
    Args:
        model (tf.keras.Model): Trained TensorFlow Keras model to evaluate.
        input_data (np.ndarray): Input voltage data for evaluation.
        image_labels (np.ndarray): Ground truth labelled images.
        threshold (float): Threshold to binarize the model's output.
    Returns:
        float: CNR value between the two images.
    """
    res = model.predict(input_data)
    binary_reconstruction = np.where(res >= threshold, 1, 0)

    signal_region = image_labels[binary_reconstruction == 1]
    background_region = image_labels[binary_reconstruction == 0]

    mean_signal = np.mean(signal_region)
    mean_background = np.mean(background_region)
    std_background = np.std(background_region)

    cnr_value = np.abs(mean_signal - mean_background) / (std_background + 1e-8)  # Avoid division by zero

    return cnr_value

def compute_SSIM_batch(
    model: tf.keras.Model,
    input_data: np.ndarray,
    image_labels: np.ndarray,
    threshold: float = 0.5,
    use_threshold_for_prediction: bool = False,
    reduction: str = "mean"  # "mean" or "none"
) -> Union[float, np.ndarray]:
    """
    Compute SSIM for a batch of images produced by `model`.

    Args:
        model: Keras model used for prediction.
        input_data: input tensor/array shaped (batch, ...).
        image_labels: ground-truth images shaped (batch, H, W) or (batch, H, W, 1).
        threshold: if `use_threshold_for_prediction` True, binarize predictions at this value.
        use_threshold_for_prediction: if True apply thresholding to predictions before SSIM.
                                      If False, use raw model outputs (recommended).
        reduction: 'mean' to return average SSIM (float), 'none' to return per-image SSIM (np.ndarray).
    Returns:
        mean SSIM (float) if reduction == 'mean', else np.ndarray with shape (batch,)
    """

    # Predict (returns shape (batch, H, W) or (batch, H, W, 1) depending on model)
    preds = model.predict(input_data)

    # Ensure arrays (numpy) and float32 dtype
    preds = np.asarray(preds, dtype=np.float32)
    labels = np.asarray(image_labels, dtype=np.float32)

    # If label images are (batch, H, W), add channel dim
    if labels.ndim == 3:
        labels = np.expand_dims(labels, -1)
    # If preds are (batch, H, W), add channel dim
    if preds.ndim == 3:
        preds = np.expand_dims(preds, -1)

    # Optionally threshold predictions to binary {0,1}
    if use_threshold_for_prediction:
        preds = (preds >= threshold).astype(np.float32)

    # Ensure both are in same value range: SSIM needs max_val param below.
    # Here we assume labels are 0/1 (binary). If they are 0..255, scale to 0..1 first.
    # If needed, normalize:
    # if labels.max() > 1.0:
    #     labels = labels / 255.0
    # if preds.max() > 1.0:
    #     preds = preds / 255.0

    # Convert to tf tensors
    t_preds = tf.convert_to_tensor(preds, dtype=tf.float32)
    t_labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    # max_val should match the value range of images (1.0 for binary/normalized)
    max_val = 1.0

    # Compute SSIM: returns shape (batch,)
    ssim_per_image = tf.image.ssim(t_labels, t_preds, max_val=max_val).numpy()

    if reduction == "mean":
        return float(np.mean(ssim_per_image))
    else:
        return ssim_per_image