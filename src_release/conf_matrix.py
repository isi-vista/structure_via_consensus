import numpy as np
from sklearn.metrics import confusion_matrix

class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=255):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.conf_mat = None
        
        self.cc_gt = 0.0
        self.cc_pred = 0.0
        self.err_l1_img_acc = 0.0
        self.err_samples = 0.0

    def update_cc(self,cur_cc_gt,cur_cc_pred,err):
        self.cc_gt+=cur_cc_gt
        self.cc_pred+=cur_cc_pred
        self.err_l1_img_acc+=err.sum()
        self.err_samples+=err.size
            
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.

        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        curr_conf_mat = confusion_matrix(y_true=ground_truth,
                                         y_pred=prediction,
                                         labels=self.labels)
        
        if self.conf_mat is not None:
            self.conf_mat += curr_conf_mat
        else:
            self.conf_mat = curr_conf_mat
    
    def IoU(self) -> tuple:
        intersection = np.diag(self.conf_mat)
        ground_truth_set = self.conf_mat.sum(axis=1)
        predicted_set = self.conf_mat.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        IoUs = intersection / union.astype(np.float32)
        mean_IoU = np.mean(IoUs)
        return mean_IoU, IoUs

    def pixel_accuracy(self) -> float:
        # extract the number of correct guesses from the diagonal
        preds_correct = np.diag(self.conf_mat)
        # extract the number of total values per class from ground truth
        trues = np.sum(self.conf_mat, axis=1)
        # calculate the total accuracy
        return np.sum(preds_correct) / np.sum(trues)

    def recall(self) -> tuple:
        tps = np.diag(self.conf_mat)
        # this below gets the tp and all the false negative for each class
        tp_fn = np.sum(self.conf_mat, axis=1)
        recalls = tps/tp_fn.astype(np.float32)
        return np.mean(recalls), recalls
    
    def precision(self) -> tuple:
        tps = np.diag(self.conf_mat)
        # this below gets the tp and all the false positive for each class
        tp_fp = np.sum(self.conf_mat, axis=0)
        precision = tps/tp_fp.astype(np.float32)
        return np.mean(precision), precision    
    
    def f_measure(self) -> tuple:
        ## each tuple index each class
        ##0 is bg
        ##1 is face
        ##2 is hair
        ##F1 score defined as https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html?highlight=f1#sklearn.metrics.f1_score
        _,recalls = self.recall()
        _,precisions = self.precision()
        f_measure = 2*(precisions*recalls)/(precisions+recalls)
        return np.mean(f_measure), f_measure  

    def err_l1_cc(self) -> float:
        err_global = abs(self.cc_gt-self.cc_pred)/self.cc_gt
        err_img_avg = self.err_l1_img_acc/self.err_samples
        return err_global,err_img_avg

