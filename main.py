import tensorflow as tf
import os
import time
from models.model_builder import Model_builder
from config.configuration import Configuration
from utils.symbol_builder import Symbol_Builder, Statistics
from utils.utils import Model_IO, TB_Builder, Early_Stopping, confm_metrics2image
from utils.data_loader import Data_loader, Preprocess_IO 
from metrics.loss import LossHandler
from metrics.metrics import Compute_statistics
from utils.optimizer_builder import Optimizer_builder
import argparse
import numpy as np
import skimage.io as io 

'''def Train(cf,sess, model,train_op,loss_fun,summary_op,summary_writer,
            saver,mean_IoU, update_IoU, running_vars_initializer):'''
def Train(cf, sess, sb, saver):
    #Path definitions
    train_image_path = os.path.join(cf.train_dataset_path, cf.train_folder_names[0])
    train_gt_path = os.path.join(cf.train_dataset_path, cf.train_folder_names[1])
    valid_image_path = os.path.join(cf.valid_dataset_path, cf.valid_folder_names[0])
    valid_gt_path = os.path.join(cf.valid_dataset_path, cf.valid_folder_names[1])
    trainable_var = tf.trainable_variables()
    # Training dataset set up
    train_set = Data_loader(cf, train_image_path, train_gt_path, cf.train_samples)
    train_set.Load_dataset(cf.train_batch_size)
    # Validation dataset set up
    valid_set = Data_loader(cf, valid_image_path, valid_gt_path, cf.valid_samples_epoch)
    valid_set.Load_dataset(cf.valid_batch_size)
    # Simbol creation for metrics and statistics
    train_stats = Statistics(cf.train_batch_size, sb)
    valid_stats = Statistics(cf.valid_batch_size, sb)
    # More summary information to add
    #tf.summary.scalar("Mean_loss", train_mLoss)
    #img_conf_mat = tf.placeholder(tf.uint8, shape=[None, 480, 640, 3], name="conf_mat")
    tf.summary.scalar("Mean_IoU", train_stats.mean_IoU)
    tf.summary.scalar("Mean_Acc", train_stats.accuracy_class)

    # Early stopping
    if cf.early_stopping:
        e_stop = Early_Stopping(cf.patience)
    # Training  
    feed_dict = []
    stop = False
    epoch = 1

    # Epoch loop
    while epoch < cf.epochs+1 and not stop:
        if cf.shuffle:
            train_set.Shuffle()
            valid_set.Shuffle()
        epoch_time = time.time()
        loss_per_batch = np.zeros(train_set.num_batches, dtype=np.float32)
        conf_mat = np.zeros((cf.num_classes,cf.num_classes), dtype=np.float32)
        # initialize/reset the running variables
        sess.run(train_stats.running_vars_initializer)
        #Dataset batch loop
        for i in range(train_set.num_batches):
            batch_x, batch_y = train_set.Next_batch(cf.train_batch_size)
            feed_dict = {sb.model.simb_image: batch_x, sb.model.simb_gt: batch_y, 
                                sb.model.simb_is_training: True}
            simbol_list = [sb.train_op, sb.loss_fun, sb.model.annotation_pred, 
                        train_stats.update_IoU, train_stats.update_acc_class, train_stats.conf_matrix_batch]
            sess_return = sess.run(simbol_list, feed_dict)
            loss_per_batch[i] = sess_return[1]
            pred = sess_return[2]
            conf_mat += sess_return[4]
        # Epoch train summary info
        conf_mat = conf_mat/train_set.num_batches
        img_conf_mat = confm_metrics2image(conf_mat)
        img_conf_mat = tf.expand_dims(img_conf_mat, 0)
        tf.summary.image("conf_mat", img_conf_mat, max_outputs=2)
        train_mLoss = np.mean(np.asarray(loss_per_batch))
        summary_op_train = sb.tensorBoard.set_up()
        mIoU_train, mAcc_train, summary_train = sess.run([train_stats.mean_IoU, 
                                                train_stats.accuracy_class, summary_op_train], feed_dict)
        train_set.Reset_Offset()

        # Validation in train
        if cf.valid_samples_epoch > 0:
            conf_mat = np.zeros((cf.num_classes,cf.num_classes), dtype=np.float32)
            valid_loss_batch = np.zeros(valid_set.num_batches, dtype=np.float32)
            sess.run(valid_stats.running_vars_initializer)
            for i in range(valid_set.num_batches):
                batch_x, batch_y = valid_set.Next_batch(cf.valid_batch_size)
                feed_dict = {sb.model.simb_image: batch_x, sb.model.simb_gt: batch_y, 
                                        sb.model.simb_is_training: False}
                simbol_list = [sb.loss_fun, sb.model.annotation_pred, valid_stats.update_IoU, 
                            valid_stats.update_acc_class, valid_stats.conf_matrix_batch]
                sess_return = sess.run(simbol_list, feed_dict)
                valid_loss_batch[i] = sess_return[0]
                pred = sess_return[1]
                conf_mat += sess_return[4]
            conf_mat = conf_mat/train_set.num_batches
            img_conf_mat = confm_metrics2image(conf_mat)
            img_conf_mat = tf.expand_dims(img_conf_mat, 0)
            summary_conf_mat_val = tf.summary.image("conf_mat_validation", img_conf_mat, max_outputs=2)
            mIoU_valid, mAcc_valid, sammary_val = sess.run([valid_stats.mean_IoU, valid_stats.accuracy_class, 
                                    summary_conf_mat_val])
            valid_mLoss = np.mean(np.asarray(valid_loss_batch))
            valid_set.Reset_Offset()

        # Screen display
        sb.tensorBoard.summary_writer.add_summary(summary_train, epoch)
        sb.tensorBoard.summary_writer.add_summary(sammary_val, epoch)
        epoch_time = time.time() - epoch_time
        print("Epoch: %d, Time: %ds \n\t Train_loss: %g, mIoU: %g, mAcc: %g" % (epoch, epoch_time,
                                train_mLoss, mIoU_train, mAcc_train))
        if cf.valid_samples_epoch > 0:
            print("\t Valid_loss: %g, mIoU: %g, mAcc: %g" % (valid_mLoss, mIoU_valid, mAcc_valid))
            saver.Save(cf, sess, train_mLoss, mIoU_train, mAcc_train, valid_mLoss, mIoU_valid, mAcc_valid)
            if cf.early_stopping:
                stop = e_stop.Check(cf.save_condition, train_mLoss, mIoU_train, mAcc_train, 
                                                    valid_mLoss, mIoU_valid, mAcc_valid)
        else:
            saver.Save(cf, sess, train_mLoss, mIoU_train, mAcc_train)
            if cf.early_stopping:
                stop = e_stop.Check(cf.save_condition, train_mLoss, mIoU_train, mAcc_train) 
        epoch += 1

def Validation(cf, sess, sb):
    valid_image_path = os.path.join(cf.valid_dataset_path, cf.valid_folder_names[0])
    valid_gt_path = os.path.join(cf.valid_dataset_path, cf.valid_folder_names[1])
    valid_set = Data_loader(cf, valid_image_path, valid_gt_path, cf.valid_samples)
    valid_set.Load_dataset(cf.valid_batch_size)
    valid_stats = Statistics(cf.valid_batch_size, sb)
    valid_loss_batch = np.zeros(valid_set.num_batches, dtype=np.float32)
    sess.run(valid_stats.running_vars_initializer)
    for i in range(valid_set.num_batches):
        batch_x, batch_y = valid_set.Next_batch(cf.valid_batch_size)
        feed_dict = {sb.model.simb_image: batch_x, sb.model.simb_gt: batch_y, 
                        sb.model.simb_is_training: False}
        simbol_list = [sb.loss_fun, sb.model.annotation_pred, valid_stats.update_IoU, 
                    valid_stats.update_acc_class]
        sess_return = sess.run(simbol_list, feed_dict)
        valid_loss_batch[i] = sess_return[0]
        pred = sess_return[1]
        conf_mat = sess_return[3]
    mIoU_valid, mAcc_valid = sess.run([valid_stats.mean_IoU, valid_stats.accuracy_class])
    print("\t Valid_loss: %g, mIoU: %g, mAcc: %g" % (np.mean(np.asarray(valid_loss_batch)),
                                                    mIoU_valid, mAcc_valid))

def Test(cf, sess, sb):
    test_image_path = os.path.join(cf.test_dataset_path, cf.test_folder_names[0])
    test_gt_path = os.path.join(cf.test_dataset_path, cf.test_folder_names[1])
    test_set = Data_loader(cf, test_image_path, test_gt_path, cf.test_samples)
    test_set.Load_dataset(cf.test_batch_size)
    test_stats = Statistics(cf.test_batch_size, sb)
    test_loss_batch = np.zeros(test_set.num_batches, dtype=np.float32)
    sess.run(test_stats.running_vars_initializer)
    for i in range(test_set.num_batches):
        batch_x, batch_y = test_set.Next_batch(cf.test_batch_size)
        feed_dict = {sb.model.simb_image: batch_x, sb.model.simb_gt: batch_y, 
                        sb.model.simb_is_training: False}
        simbol_list = [sb.loss_fun, sb.model.annotation_pred, test_stats.update_IoU, 
                    test_stats.update_acc_class]
        sess_return = sess.run(simbol_list, feed_dict)
        test_loss_batch[i] = sess_return[0]
        pred = sess_return[1]
        conf_mat = sess_return[3]
    mIoU_test, mAcc_test = sess.run([test_stats.mean_IoU, test_stats.accuracy_class])
    print("\t test_loss: %g, mIoU: %g, mAcc: %g" % (np.mean(np.asarray(test_loss_batch)),
                                                    mIoU_test, mAcc_test))

def main():
    start_time = time.time()
    # Input arguments
    parser = argparse.ArgumentParser(
        description="TensorFlow framework for Semantic Segmentation")

    parser.add_argument("--config_file",
                        type=str,
                        default='config/configFile.py',
                        help="configuration file path")

    parser.add_argument("--exp_name",
                        type=str,
                        default='Sample',
                        help="Experiment name")

    parser.add_argument("--exp_folder",
                        type=str,
                        default='/home/jlgomez/Experiments/DenseNetFCN/',
                        help="Experiment folder path")

    args = parser.parse_args()

    # Prepare configutation
    print ('Loading configuration ...')
    config = Configuration(args.config_file, args.exp_name, args.exp_folder)
    cf = config.Load()
    
    #Create symbol builder with all the parameters needed (model, loss, optimizers,...)
    sb = Symbol_Builder(cf)

    # TensorFlow session
    print ('Starting session ...')
    sess = tf.Session()
    saver = Model_IO()
    #Saver Log for TesnorBoard
    sb.tensorBoard.save(cf.exp_folder + cf.log_path, sess)    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore session
    if cf.load_model == 'tensorflow':
        print ('Loading model ...')
        saver.Load_model(cf, sess)
    elif cf.load_model == 'keras':
        print ('Loading weights ...')
        saver.Manual_weight_load(cf, sess)
    '''start_step = sess.run(model)
    sess.run(tf.assign(model, start_step))'''
    '''saver = tf.train.import_meta_graph('tensorflow_model/my_test_model-1000')
    saver.restore(sess,tf.train.latest_checkpoint('tensorflow_model/'))'''
    
    # training step
    if cf.train:
        print ('Starting training ...')
        Train(cf,sess,sb,saver)
    if cf.validation:
        print ('Starting validation ...')
        Validation(cf, sess, sb)
    if cf.test:
        print ('Starting testing ...')
        Test(cf, sess, sb)
    total_time = time.time() - start_time    
    print(' Experiment finished: %ds ' % (total_time))

# Entry point of the script
if __name__ == "__main__":
    main()