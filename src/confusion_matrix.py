class ConfusionMatrix:
    def __init__(self, tp, fp, fn, tn):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def __repr__(self):
        return '\n' + str(self.tp) + '\t' + str(self.fp) + '\n' +\
               str(self.fn) + '\t' + str(self.tn) + '\n'

    def get_accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total

    def get_error(self):
        return 1 - self.get_accuracy()

    def get_sensitivity(self):
        return self.tp / (self.tp + self.fn)

    def get_specificity(self):
        return self.tn / (self.tn + self.fp)

    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        return self.get_sensitivity()