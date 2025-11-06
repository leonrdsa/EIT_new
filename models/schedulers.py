import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class SchedulerandTrackerCallback(Callback):
    def __init__(self, scheduler=None):
        super().__init__()
        self.scheduler = scheduler
        self.epoch_lr = []
        self.epoch_loss = []
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Get current learning rate from the schedule
        if hasattr(self.model.optimizer, '_decayed_lr'):
            # For schedulers that use tf.keras.optimizers.schedules
            current_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        else:
            # Fallback for constant learning rate
            current_lr = self.model.optimizer.learning_rate.numpy()
            
        loss = logs.get('loss')
        self.epoch_lr.append(current_lr)
        self.epoch_loss.append(loss)