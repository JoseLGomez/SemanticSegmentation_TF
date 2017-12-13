import os
import tensorflow as tf
import numpy as np
from utils import Model_IO, TB_Builder
from data_loader import Data_loader, Preprocess_IO 
from optimizer_builder import Optimizer_builder 
from metrics.loss import LossHandler
from metrics.metrics import Compute_statistics
from models.model_builder import Model_builder


class Symbol_Builder():
	def __init__(self, cf):
		self.cf = cf
		self.__build_model()
		self.trainable_var = tf.trainable_variables()
		self.__build_loss()
		self.__build_TB()
		self.__build_optimizer()

	def __build_model(self):
		# Prepare model
	    print ('Generating model ...')
	    self.model = Model_builder(self.cf)
	    self.model.Build()
    
	def __build_loss(self):
	    # Prepare Loss
	    print ('Configuring metrics ...')
	    self.loss = LossHandler(self.cf)
	    self.loss_fun = self.loss.loss(self.model.logits, self.model.simb_gt, self.trainable_var)

	def __build_TB(self):
	    # Prepare TensorBoard
	    print ('Preparing TensorBoard ...')
	    self.tensorBoard = TB_Builder(self.cf, self.model, self.loss_fun)
    
	def __build_optimizer(self):
	    # Prepare optimizer
	    print ('Selecting optimizer ...')
	    self.optimizer = Optimizer_builder().build(self.cf)
	    self.grads_and_vars, self.train_op = Optimizer_builder().minimize(self.optimizer,
	    														self.loss_fun,self.trainable_var)

class Statistics():
	def __init__(self,batch_size,sb):
		self.sb = sb
		self.num_classes = sb.cf.num_classes
		self.batch_size = batch_size
		self.__build_statistics()

	def __build_statistics(self):
		self.labels = tf.reshape(self.sb.model.simb_gt, [self.batch_size,-1])
		self.predictions = tf.reshape(self.sb.model.annotation_pred, [self.batch_size,-1])
		#compute mIoU
		with tf.name_scope("Statistics"):
			self.mean_IoU, self.update_IoU = tf.metrics.mean_iou(labels=self.labels, 
																predictions=self.predictions,
																num_classes=self.num_classes, 
																weights=None, name='IoU_metric')
			self.accuracy_class, self.update_acc_class = tf.metrics.mean_per_class_accuracy(
															labels=self.labels, 
															predictions=self.predictions,
															num_classes=self.num_classes, 
															weights=None, 
															name='Acc_Class_metric') 
			# Compute TP
			self.TP, self.update_TP = tf.metrics.true_positives(self.labels, self.predictions, name='TP')
			# Compute FN
			self.FN, self.update_FN = tf.metrics.false_negatives(self.labels, self.predictions, name='FN')
			# Compute FP
			self.FP, self.update_FP = tf.metrics.false_positives(self.labels, self.predictions, name='FP')
			# Confusion matrix
			self.list_labels = tf.unstack(self.labels)
			self.list_preds = tf.unstack(self.predictions)
			self.conf_matrix_batch = np.zeros((self.num_classes,self.num_classes), dtype=np.float32)
			for img in range(self.batch_size):
				conf_matrix = tf.confusion_matrix(self.list_labels[img], self.list_preds[img], 
														self.num_classes)
				#self.conf_matrix_batch.append(conf_matrix)
				self.conf_matrix_batch += conf_matrix
			self.conf_matrix_batch = self.conf_matrix_batch/self.batch_size
		# Define initializer to initialize/reset running variables
		self.local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='Statistics')
		self.running_vars_initializer = tf.variables_initializer(var_list=self.local_vars)
