import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import cv2 as cv
import matplotlib
import StringIO
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def Print_Accuracy_per_class(acc_per_class):
    pass

def confm_metrics2image(conf_matrix,names=None):
    nLabels = np.shape(conf_matrix)[0]

    if names==None:
        plt_names = range(nLabels)
    else:
        plt_names = names

    conf_matrix = np.asarray(conf_matrix, dtype=np.float32)

    for i in range(nLabels):
        sum_row = sum(conf_matrix[i][:])
        for j in range(nLabels):
            if sum_row == 0:
                conf_matrix[i][j] = 0
            else:
                conf_matrix[i][j] = (conf_matrix[i][j]) / float(sum_row)

    img = StringIO.StringIO()
    plt.ioff()
    plt.cla()
    plt.clf()
    plt.imshow(conf_matrix,
               interpolation='nearest',
               cmap=plt.cm.Blues,
               vmin=0.0,
               vmax=1.0)
    plt.colorbar()
    plt.title('Confusion Matrix')

    plt.xticks(range(nLabels),plt_names, rotation=90)
    ystick = zip(plt_names, [conf_matrix[i][i] for i in range(nLabels)])
    ystick_str = [str(ystick[i][0]) + '(%.2f)' % ystick[i][1] for i in range(nLabels)]

    plt.yticks(range(nLabels), ystick_str)

    plt.xlabel('Prediction Label')
    plt.ylabel('True Label')

    plt.draw()
    plt.pause(0.1)
    plt.savefig(img, format='png')
    img.seek(0)

    data = np.asarray(bytearray(img.buf), dtype=np.uint8)
    img = cv.imdecode(data, cv.IMREAD_UNCHANGED)[:, :, 0:3]
    img = img[..., ::-1]

    return img

def save_prediction(output_path, predictions, names):
    for img in range(len(names)):
        output_file = output_path + names[img]
        cv.imwrite(output_file, np.squeeze(predictions[img], axis=2))

class TB_Builder():
    def __init__(self, cf, model, loss):
        tf.summary.image("input_image", model.simb_image, max_outputs=2)
        tf.summary.image("ground_truth", tf.cast(model.simb_gt, tf.uint8), max_outputs=2)
        tf.summary.image("pred_annotation", tf.cast(model.annotation_pred, tf.uint8), max_outputs=2)
        tf.summary.scalar("Loss", loss)
        
    def set_up(self, key=None):
        if key is not None:
            return tf.summary.merge_all(key=key)
        else:    
            return tf.summary.merge_all()

    def save(self, path, sess):
        return tf.summary.FileWriter(path, sess.graph)

class Early_Stopping():
    def __init__(self, patience):
        self.best_loss_metric = float('inf')
        self.best_metric = 0
        self.counter = 0
        self.patience = patience
        self.stop = False

    def Check(self, save_condition, train_mLoss, mIoU_train, mAcc_train, valid_mLoss=None, 
                mIoU_valid=None, mAcc_valid=None):
        if save_condition == 'train_loss':
            if train_mLoss < self.best_loss_metric:
                self.best_loss_metric = train_mLoss
                self.counter = 0
            else:
                self.counter += 1
        elif save_condition == 'train_mIoU':
            if mIoU_train > self.best_metric:
                self.best_metric = mIoU_train
                self.counter = 0
            else:
                self.counter += 1    
        elif save_condition == 'train_mAcc':
            if mAcc_train > self.best_metric:
                self.best_metric = mAcc_train
                self.counter = 0
            else:
                self.counter += 1
        elif save_condition == 'valid_loss':
            if valid_mLoss < self.best_loss_metric:
                self.best_loss_metric = valid_mLoss
                self.counter = 0
            else:
                self.counter += 1
        elif save_condition == 'valid_mIoU':
            if mIoU_valid > self.best_metric:
                self.best_metric = mIoU_valid
                self.counter = 0
            else:
                self.counter += 1
        elif save_condition == 'valid_mAcc':
            if mAcc_valid > self.best_metric:
                self.best_metric = mAcc_valid
                self.counter = 0
            else:
                self.counter += 1
        if self.counter == self.patience:
            return True
        else:
            return False