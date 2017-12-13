import tensorflow as tf
import os
import numpy as np

def Compute_statistics(labels, predictions, num_classes, batch_size, weights=None):
	labels = tf.reshape(labels, [batch_size,-1])
	predictions = tf.reshape(predictions, [batch_size,-1])
	mean_IoU, update_IoU = tf.metrics.mean_iou(labels=labels, predictions=predictions,
						num_classes=num_classes, weights=weights, name='IoU_metric')
	#conf_matrix = tf.confusion_matrix(labels, predictions, num_classes)
	TP, update_TP = tf.metrics.true_positives(labels, predictions)
	FN, update_FN = tf.metrics.false_negatives(labels, predictions)
	FP, update_FP = tf.metrics.false_positives(labels, predictions)
	#return mean_IoU, TP, FN, FP
	return mean_IoU, update_IoU