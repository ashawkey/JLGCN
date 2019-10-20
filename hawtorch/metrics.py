import os
import torch
import numpy as np

class IoUAverager:
    def __init__(self, nCls, eps=1e-5):
        self.nCls = nCls
        self.eps = eps
        self.shape_ious = [[] for _ in range(self.nCls)]

    def clear(self):
        self.shape_ious = [[] for _ in range(self.nCls)]

    def update(self, outputs, truths):
        preds = outputs.max(dim=1)[1]
        preds_np = preds.detach().cpu().numpy()
        pids_np = truths.detach().cpu().numpy()

        batch_size = pids_np.shape[0]
        for batch in range(batch_size):
            for part in range(self.nCls):
                I = np.sum(np.logical_and(preds_np[batch] == part, pids_np[batch] == part))
                U = np.sum(np.logical_or(preds_np[batch] == part, pids_np[batch] == part))
                if U == 0: continue
                else: self.shape_ious[part].append(I/U)

    def measure(self):
        res = []
        for part in range(self.nCls):
            if self.shape_ious[part] != []:
                res.append(np.mean(self.shape_ious[part]))
        return np.mean(res)

    def better(self, A, B):
        return A > B

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "mIoU"), self.measure(), global_step)

    def report(self):
        text = f"mIoU = {self.measure():.4f}\n"
        for part in range(self.nCls):
            if self.shape_ious[part] != []:
                text += f"\t Class {part}: {np.mean(self.shape_ious[part]):.4f}\n"
            else:
                text += f"\t Class {part}: None\n"
        return text


class ClassificationAverager:
    """ statistics for classification """
    def __init__(self, nCls, eps=1e-5, names=None):
        self.nCls = nCls
        self.names = names
        self.eps = eps
        self.N = 0
        self.table = np.zeros((self.nCls, 4), dtype=np.int32)
        self.hist_preds = []
        self.hist_truths = []

    def clear(self):
        self.N = 0
        self.table = np.zeros((self.nCls, 4), dtype=np.int32)
        self.hist_preds = []
        self.hist_truths = []

    def update(self, outputs, truths):
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy() # [B, ]
        labels = truths.detach().cpu().numpy() # [B, ]

        self.hist_preds.extend(preds.tolist())
        self.hist_truths.extend(labels.tolist())

        self.N += np.prod(labels.shape)
        for Cls in range(self.nCls):
            true_positive = np.count_nonzero(np.bitwise_and(preds == Cls, labels == Cls))
            true_negative = np.count_nonzero(np.bitwise_and(preds != Cls, labels != Cls))
            false_positive = np.count_nonzero(np.bitwise_and(preds == Cls, labels != Cls))
            false_negative = np.count_nonzero(np.bitwise_and(preds != Cls, labels == Cls))
            self.table[Cls] += [true_positive, true_negative, false_positive, false_negative]

    def measure(self):
        """Overall Accuracy"""
        total_TP = np.sum(self.table[:, 0]) # all true positives 
        accuracy = total_TP/self.N
        return accuracy

    def better(self, A, B):
        return A > B

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "Accuracy"), self.measure(), global_step)
    
    def plot_conf_mat(self):
        #mat = confusion_matrix(self.hist_truths, self.hist_preds)
        from .vision import plot_confusion_matrix
        plot_confusion_matrix(self.hist_truths, self.hist_preds)

    def report(self, each_class=False, conf_mat=False):
        precisions = []
        recalls = []
        for Cls in range(self.nCls):
            precision = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,3] + self.eps) # TP / (TP + FN)
            recall = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,2] + self.eps) # TP / (TP + FP)
            precisions.append(precision)
            recalls.append(recall)
        total_TP = np.sum(self.table[:, 0]) # all true positives 
        accuracy = total_TP/self.N
        accuracy_mean_class = np.mean(precisions)

        text = f"Overall Accuracy = {accuracy:.4f}({total_TP}/{self.N})\n"
        text += f"\tMean-class Accuracy = {accuracy_mean_class:.4f}\n"
        
        if each_class:
            for Cls in range(self.nCls):
                if precisions[Cls] != 0 or recalls[Cls] != 0:
                    text += f"\tClass {str(Cls)+'('+self.names[Cls]+')' if self.names is not None else Cls}: precision = {precisions[Cls]:.3f} recall = {recalls[Cls]:.3f}\n"
        if conf_mat:
            self.plot_conf_mat()

        return text
