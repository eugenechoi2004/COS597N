import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score, accuracy_score

class Metrics():

    def __init__(self, metric_path):
        self.training_loss = []
        self.validation_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []
        self.training_f1 = []
        self.validation_f1 = []
        self.learning_rates = []
        self.epoch_times = []
        self.metric_path = metric_path

    def track_loss_point(self, train_loss, val_loss):
        self.training_loss.append(train_loss)
        self.validation_loss.append(val_loss)

    def track_accuracy_point(self, train_acc, val_acc):
        self.training_accuracy.append(train_acc)
        self.validation_accuracy.append(val_acc)

    def track_f1_point(self, train_f1, val_f1):
        self.training_f1.append(train_f1)
        self.validation_f1.append(val_f1)

    def track_learning_rate(self, lr):
        self.learning_rates.append(lr)

    def start_epoch_timer(self):
        self.start_time = time.time()

    def end_epoch_timer(self):
        self.epoch_times.append(time.time() - self.start_time)

    def save_loss_plot(self):
        epochs = range(1, len(self.training_loss) + 1)
        plt.plot(epochs, self.training_loss, 'b', label='Training Loss')
        plt.plot(epochs, self.validation_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.metric_path}_loss.png")
        plt.close()

    def save_accuracy_plot(self):
        epochs = range(1, len(self.training_accuracy) + 1)
        plt.plot(epochs, self.training_accuracy, 'b', label='Training Accuracy')
        plt.plot(epochs, self.validation_accuracy, 'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f"{self.metric_path}_accuracy.png")
        plt.close()