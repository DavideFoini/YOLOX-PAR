class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.5):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else: self.counter += 0

class OverfittingAcc:
    def __init__(self, tolerance=10):

        self.tolerance = tolerance
        self.counter = 0
        self.stop = False
        self.best_accuracy = -1

    def __call__(self, last_acc):
        if last_acc > self.best_accuracy:
            self.best_accuracy = last_acc
            self.counter = 0
        else: self.counter += 1

        if self.counter == self.tolerance: self.stop = True

class OverfittingLoss:
    def __init__(self, tolerance=10):

        self.tolerance = tolerance
        self.counter = 0
        self.stop = False
        self.best = True
        self.best_loss = float('inf')

    def __call__(self, last_loss):
        delta = self.best_loss - last_loss
        if last_loss < self.best_loss and delta > 0.005:
            self.best_loss = last_loss
            self.counter = 0
            self.best = True
        else:
            self.counter += 1
            self.best = False

        print("Overfitting counter: {}".format(self.counter))
        if self.counter == self.tolerance: self.stop = True