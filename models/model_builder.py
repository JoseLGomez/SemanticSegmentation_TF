import tensorflow as tf
import os


from denseNetTensorFlow import DenseNetFCN

class Model_builder():
    def __init__(self, cf):
        self.simb_image = tf.placeholder(tf.float32, shape=[None, cf.resize_image[0], 
                                cf.resize_image[1], cf.image_channels], name="input_image")
        self.simb_gt = tf.placeholder(tf.int32, shape=[None, cf.resize_image[0], 
                                cf.resize_image[1], 1], name="gt_image")
        self.simb_is_training = tf.placeholder(tf.bool, name="is_training")
        self.cf = cf
        self.logits = None
        self.annotation_pred = None
        
    def Build(self):       
        if self.cf.model_type == 'DenseNetFCN':
            model = DenseNetFCN(self.simb_image, nb_dense_block=self.cf.model_blocks, 
                                growth_rate=self.cf.model_growth, nb_layers_per_block=self.cf.model_layers,
                                upsampling_type=self.cf.model_upsampling,
                                classes=self.cf.num_classes, is_training=self.simb_is_training)
        else:
            raise ValueError('Unknown model')
        self.annotation_pred = tf.argmax(model, axis=3, name="prediction")
        self.annotation_pred = tf.expand_dims(self.annotation_pred, dim=3)
        self.logits = model