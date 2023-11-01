import matplotlib.pyplot as plt


class Metrics():

    self.training_loss = []
    self.validation_loss = []
    self.metric_path = ""

    def __init__(self, metric_path):
        self.metric_path = metric_path

    def track_loss_point(self, train_loss_point, val_loss_point):
        self.training_loss.append(train_loss_point)
        self.validation_loss.append(val_loss_point)

    def save_loss(self):
        epochs = range(1, len(self.training_loss) + 1)
        plt.plot(epochs, self.training_loss, 'b', label='Training Loss')
        plt.plot(epochs, self.validation_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.metric_path)
        plt.close()


