import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow


from denseNetTensorFlow import DenseNetFCN

class Model_builder():
    def __init__(self, cf, size):
        '''self.simb_image = tf.placeholder(tf.float32, shape=[None, size[0], 
                                size[1], cf.image_channels], name="input_image")
        self.simb_gt = tf.placeholder(tf.int32, shape=[None, size[0], 
                                size[1], 1], name="gt_image")'''
        self.simb_image = tf.placeholder(tf.float32, shape=[None, None, 
                                None, cf.image_channels], name="input_image")
        self.simb_gt = tf.placeholder(tf.int32, shape=[None, None, 
                                None, 1], name="gt_image")
        self.simb_is_training = tf.placeholder(tf.bool, name="is_training")
        self.cf = cf
        self.logits = None
        self.annotation_pred = None
        
    def Build(self, sess):
        self.modelIO = Model_IO()
        if self.cf.pretrained_model and not self.cf.weight_only:
            if self.cf.model_type == 'DenseNetFCN':
                if self.cf.load_model == 'tensorflow':
                    self.simb_image, model = self.modelIO.Load_model(self.cf, sess)
                elif self.cf.load_model == 'keras':
                    print ('Loading model from keras...')
                    self.simb_image, model, self.simb_is_training = self.modelIO.Load_keras_model(self.cf, sess)
            else:
                raise ValueError('Unknown model')
        else:
            with tf.name_scope(self.cf.model_name):
                if self.cf.model_type == 'DenseNetFCN':
                    model = DenseNetFCN(self.simb_image, nb_dense_block=self.cf.model_blocks, 
                                    growth_rate=self.cf.model_growth, nb_layers_per_block=self.cf.model_layers,
                                    upsampling_type=self.cf.model_upsampling,
                                    classes=self.cf.num_classes, is_training=self.simb_is_training)
                else:
                    raise ValueError('Unknown model')
            if self.cf.pretrained_model and self.cf.weight_only:
                print ('Loading weights ...')
                if self.cf.load_model == 'tensorflow':
                    self.modelIO.Load_weights(self.cf, sess)
                elif self.cf.load_model == 'keras':
                    print ('Loading weights from keras model...')
                    self.modelIO.Manual_weight_load(self.cf, sess)                       
        
        self.annotation_pred = tf.argmax(model, axis=3, name="prediction")
        self.annotation_pred = tf.expand_dims(self.annotation_pred, dim=3)
        self.logits = model

class Model_IO():
    def __init__(self):
        self.train_mLoss = float('inf')
        self.mIoU_train = 0
        self.mAcc_train = 0
        self.valid_mLoss = float('inf')
        self.mIoU_valid = 0
        self.mAcc_valid = 0
    
    def Load_keras_model(self, cf, sess):
        restorer = tf.train.import_meta_graph(os.path.join(cf.model_path, cf.model_name)+'.meta')
        restorer.restore(sess,tf.train.latest_checkpoint(cf.model_path))
        graph = tf.get_default_graph()
        input()
        variables_to_restore = [var for var in tf.global_variables()]
        variables_to_read = [var for var in tf.global_variables() if 'Adam' not in var.name]
        reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(cf.model_path, cf.model_name))
        var_to_shape_map = reader.get_variable_to_shape_map()
        restorer = tf.train.Saver(variables_to_restore)
        for var in variables_to_read: 
            print var.name
  
        bn1 = graph.get_tensor_by_name("bn1_layer1_block_down1/keras_learning_phase:0")  
        simb_image = graph.get_tensor_by_name("input_1:0")
        model = graph.get_tensor_by_name("nd_softmax_1/transpose_1:0")
        return simb_image, model, bn1

    def Load_model(self, cf, sess):

        restorer = tf.train.import_meta_graph(os.path.join(cf.model_path, cf.model_name)+'2.meta')
        for n in tf.get_default_graph().as_graph_def().node:
            print n.name 
        
        #variables_to_restore = [var for var in tf.global_variables() if 'Adam' not in var.name]
        restorer.restore(sess, os.path.join(cf.model_path, cf.model_name))
        graph = tf.get_default_graph()

        #bn1 = graph.get_tensor_by_name("bn1_layer1_block_down1/keras_learning_phase:0")  
        simb_image = graph.get_tensor_by_name("input_image:0")
        model = graph.get_tensor_by_name("Reshape_1:0")

        return simb_image, model
    
    def Load_weights(self, cf, sess):
        variables_to_restore = [var for var in tf.global_variables()]
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, os.path.join(cf.model_path, cf.model_name))
    
    def Manual_weight_load(self, cf, sess):
        variables_to_read = [var for var in tf.global_variables() if 'Adam' not in var.name]
        reader = tf.pywrap_tensorflow.NewCheckpointReader(os.path.join(cf.model_path, cf.model_name))
        var_to_shape_map = reader.get_variable_to_shape_map()
        for var in var_to_shape_map:
            if cf.model_type == 'DenseNetFCN':
                if 'kernel' in var:
                    new_var = var.replace('/kernel','/weights')
                    match_vars = [i for i in variables_to_read if new_var in i.name]
                    sess.run(match_vars[0].assign(reader.get_tensor(var)))
                elif var == 'iterations':
                    pass
                elif var == 'decay':
                    pass
                elif var == 'lr':
                    pass
                elif var == 'beta_1':
                    pass
                elif var == 'beta_2':
                    pass
                else:
                    match_vars = [i for i in variables_to_read if var in i.name]
                    if len(match_vars) > 0:
                        sess.run(match_vars[0].assign(reader.get_tensor(var)))
                    else:
                        #print var
                        raise ValueError('weight load mismatch, check if the model to preload is from the DenseNetFCN from keras')

    def Save(self, cf, sess, train_mLoss, mIoU_train, mAcc_train, valid_mLoss=None, 
                mIoU_valid=None, mAcc_valid=None):
        variables_to_save = [var for var in tf.global_variables() if 'Adam' not in var.name]
        #print variables_to_save
        saver = tf.train.Saver(variables_to_save)
        meta_graph_def = tf.train.export_meta_graph(filename=os.path.join(cf.exp_folder, cf.model_name)+'2.meta',
                                            collection_list=[cf.model_name])
        print meta_graph_def
        if cf.save_condition == 'always':
            saver.save(sess, os.path.join(cf.exp_folder, cf.model_name))
        elif cf.save_condition == 'train_loss':
            if train_mLoss < self.train_mLoss:
                self.train_mLoss = train_mLoss
                saver.save(sess, os.path.join(cf.exp_folder, cf.model_name))
        elif cf.save_condition == 'train_mIoU':
            if mIoU_train > self.mIoU_train:
                self.mIoU_train = mIoU_train
                saver.save(sess, os.path.join(cf.exp_folder, cf.model_name))
        elif cf.save_condition == 'train_mAcc':
            if mAcc_train > self.mAcc_train:
                self.mAcc_train = mAcc_train
                saver.save(sess, os.path.join(cf.exp_folder, cf.model_name))
        elif cf.save_condition == 'valid_loss':
            if valid_mLoss < self.valid_mLoss:
                self.valid_mLoss = valid_mLoss
                saver.save(sess, os.path.join(cf.exp_folder, cf.model_name))
        elif cf.save_condition == 'valid_mIoU':
            if mIoU_valid > self.mIoU_valid:
                self.mIoU_valid = mIoU_valid
                saver.save(sess, os.path.join(cf.exp_folder, cf.model_name))
        elif cf.save_condition == 'valid_mAcc':
            if mAcc_valid > self.mAcc_valid:
                self.mAcc_valid = mAcc_valid
                saver.save(sess, os.path.join(cf.exp_folder, cf.model_name))