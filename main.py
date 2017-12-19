import tensorflow as tf
import os
import time
from models.model_builder import Model_builder, Model_IO
from config.configuration import Configuration
from utils.symbol_builder import Symbol_Builder, Statistics
from utils.utils import TB_Builder, Early_Stopping, confm_metrics2image, save_prediction
from utils.data_loader import Data_loader, Preprocess_IO 
from metrics.loss import LossHandler
from metrics.metrics import Compute_statistics
from utils.optimizer_builder import Optimizer_builder
from utils.ProgressBar import ProgressBar
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
    train_set = Data_loader(cf, train_image_path, cf.train_samples, 
                                        cf.size_image_train, train_gt_path)
    train_set.Load_dataset(cf.train_batch_size)
    # Validation dataset set up
    valid_set = Data_loader(cf, valid_image_path, cf.valid_samples_epoch, 
                                        cf.size_image_valid, valid_gt_path)
    valid_set.Load_dataset(cf.valid_batch_size)
    # Simbol creation for metrics and statistics
    train_stats = Statistics(cf.train_batch_size, sb)
    valid_stats = Statistics(cf.valid_batch_size, sb)
    # More summary information to add
    #tf.summary.scalar("Mean_loss", train_mLoss)
    #img_conf_mat = tf.placeholder(tf.uint8, shape=[None, 480, 640, 3], name="conf_mat")
    tf.summary.scalar("Mean_IoU/train", train_stats.mean_IoU, collections=['train'])
    tf.summary.scalar("Mean_Acc/train", train_stats.accuracy_class, collections=['train'])
    tf.summary.scalar("Mean_IoU/train_valid", valid_stats.mean_IoU, collections=['train_valid'])
    tf.summary.scalar("Mean_Acc/train_valid", valid_stats.accuracy_class, collections=['train_valid'])

    train_writer = sb.tensorBoard.save(cf.exp_folder + cf.log_path + 'train/', sess)
    val_writer = sb.tensorBoard.save(cf.exp_folder + cf.log_path + 'train_valid/', sess)

    # Early stopping
    if cf.early_stopping:
        e_stop = Early_Stopping(cf.patience)
    # Training  
    feed_dict = []
    stop = False
    epoch = 1

    # Epoch loop
    while epoch < cf.epochs+1 and not stop:
        epoch_time = time.time()
        if cf.shuffle:
            train_set.Shuffle()
            valid_set.Shuffle()
        loss_per_batch = np.zeros(train_set.num_batches, dtype=np.float32)
        conf_mat = np.zeros((cf.num_classes,cf.num_classes), dtype=np.float32)
        # initialize/reset the running variables
        sess.run(train_stats.running_vars_initializer)
        #Progress bar
        prog_bar = ProgressBar(train_set.num_batches)
        #Dataset batch loop
        for i in range(train_set.num_batches):
            batch_x, batch_y = train_set.Next_batch(cf.train_batch_size, crop=True)
            feed_dict = {sb.model.simb_image: batch_x, sb.model.simb_gt: batch_y, 
                                sb.model.simb_is_training: True}
            simbol_list = [sb.train_op, sb.loss_fun, sb.model.annotation_pred, 
                            train_stats.update_IoU, train_stats.update_acc_class, 
                            train_stats.conf_matrix_batch]
            sess_return = sess.run(simbol_list, feed_dict)
            loss_per_batch[i] = sess_return[1]
            #pred = sess_return[2]
            conf_mat += sess_return[5]
            prog_bar.update()
        # Epoch train summary info
        conf_mat = conf_mat/train_set.num_batches
        img_conf_mat = confm_metrics2image(conf_mat, cf.labels)
        img_conf_mat = tf.expand_dims(img_conf_mat, 0)
        tf.summary.image("conf_mat/train", img_conf_mat, max_outputs=2, collections=['train'])
        train_mLoss = np.mean(np.asarray(loss_per_batch))
        summary_op_train = sb.tensorBoard.set_up('train')
        mIoU_train, mAcc_train, summary_train = sess.run([train_stats.mean_IoU, 
                                                train_stats.accuracy_class, summary_op_train], 
                                                feed_dict)
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
            img_conf_mat = confm_metrics2image(conf_mat, cf.labels)
            img_conf_mat = tf.expand_dims(img_conf_mat, 0)
            tf.summary.image("conf_mat/train_valid", 
                                        img_conf_mat, max_outputs=2, collections=['train_valid'])
            summary_op_val = sb.tensorBoard.set_up('train_valid')
            mIoU_valid, mAcc_valid, sammary_val = sess.run([valid_stats.mean_IoU, 
                                                            valid_stats.accuracy_class, 
                                                            summary_op_val])
            valid_mLoss = np.mean(np.asarray(valid_loss_batch))
            valid_set.Reset_Offset()

        # Screen display
        train_writer.add_summary(summary_train, epoch)
        val_writer.add_summary(sammary_val, epoch)
        epoch_time = time.time() - epoch_time
        print("Epoch: %d, Time: %ds \n\t Train_loss: %g, mIoU: %g, mAcc: %g" % (epoch, epoch_time,
                                train_mLoss, mIoU_train, mAcc_train))
        if cf.valid_samples_epoch > 0:
            print("\t Valid_loss: %g, mIoU: %g, mAcc: %g" % (valid_mLoss, mIoU_valid, mAcc_valid))
            saver.Save(cf, sess, train_mLoss, mIoU_train, mAcc_train, 
                                        valid_mLoss, mIoU_valid, mAcc_valid)
            if cf.early_stopping:
                stop = e_stop.Check(cf.save_condition, train_mLoss, mIoU_train, mAcc_train, 
                                                    valid_mLoss, mIoU_valid, mAcc_valid)
        else:
            saver.Save(cf, sess, train_mLoss, mIoU_train, mAcc_train)
            if cf.early_stopping:
                stop = e_stop.Check(cf.save_condition, train_mLoss, mIoU_train, mAcc_train) 
        epoch += 1

def Validation(cf, sess, sb):
    val_time = time.time()
    val_writer = sb.tensorBoard.save(cf.exp_folder + cf.log_path + 'validation/', sess)
    valid_image_path = os.path.join(cf.valid_dataset_path, cf.valid_folder_names[0])
    valid_gt_path = os.path.join(cf.valid_dataset_path, cf.valid_folder_names[1])
    valid_set = Data_loader(cf, valid_image_path, cf.valid_samples, 
                                        cf.size_image_valid, valid_gt_path)
    valid_set.Load_dataset(cf.valid_batch_size)
    valid_stats = Statistics(cf.valid_batch_size, sb)
    tf.summary.scalar("Mean_IoU/validation", valid_stats.mean_IoU, 
                                                    collections=['validation'])
    tf.summary.scalar("Mean_Acc/validation", valid_stats.accuracy_class, 
                                                    collections=['validation'])
    valid_loss_batch = np.zeros(valid_set.num_batches, dtype=np.float32)
    sess.run(valid_stats.running_vars_initializer)
    prog_bar = ProgressBar(valid_set.num_batches)
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
        prog_bar.update()
    conf_mat = conf_mat/valid_set.num_batches
    img_conf_mat = confm_metrics2image(conf_mat, cf.labels)
    img_conf_mat = tf.expand_dims(img_conf_mat, 0)
    tf.summary.image("conf_mat/validation", 
                                img_conf_mat, max_outputs=2, collections=['validation'])
    summary_op_val = sb.tensorBoard.set_up('validation')
    mIoU_valid, mAcc_valid, sammary_val = sess.run([valid_stats.mean_IoU, 
                                                valid_stats.accuracy_class, summary_op_val])
    val_time = time.time() - val_time
    print("\t Loss: %g, mIoU: %g, mAcc: %g, Time: %ds" % (np.mean(np.asarray(valid_loss_batch)),
                                                    mIoU_valid, mAcc_valid, val_time))
    val_writer.add_summary(sammary_val)

def Test(cf, sess, sb):
    test_time = time.time()
    test_writer = sb.tensorBoard.save(cf.exp_folder + cf.log_path + 'test/', sess) 
    test_image_path = os.path.join(cf.test_dataset_path, cf.test_folder_names[0])
    test_gt_path = os.path.join(cf.test_dataset_path, cf.test_folder_names[1])
    test_set = Data_loader(cf, test_image_path, cf.test_samples, 
                                    cf.size_image_test, test_gt_path)
    test_set.Load_dataset(cf.test_batch_size)
    test_stats = Statistics(cf.test_batch_size, sb)
    tf.summary.scalar("Mean_IoU/test", test_stats.mean_IoU, 
                                                    collections=['test'])
    tf.summary.scalar("Mean_Acc/test", test_stats.accuracy_class, 
                                                    collections=['test'])
    test_loss_batch = np.zeros(test_set.num_batches, dtype=np.float32)
    sess.run(test_stats.running_vars_initializer)
    prog_bar = ProgressBar(test_set.num_batches)
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
        prog_bar.update()
    conf_mat = conf_mat/test_set.num_batches
    img_conf_mat = confm_metrics2image(conf_mat, cf.labels)
    img_conf_mat = tf.expand_dims(img_conf_mat, 0)
    tf.summary.image("conf_mat/test", 
                                img_conf_mat, max_outputs=2, collections=['test'])
    summary_op_test = sb.tensorBoard.set_up('test')
    mIoU_test, mAcc_test, sammary_test = sess.run([test_stats.mean_IoU, test_stats.accuracy_class, summary_op_test])
    test_time = time.time() - test_time
    print("\t test_loss: %g, mIoU: %g, mAcc: %g, Time: %ds" % (np.mean(np.asarray(test_loss_batch)),
                                                    mIoU_test, mAcc_test, test_time))
    test_writer.add_summary(sammary_test)

def Predict(cf, sess, sb):
    predict_time = time.time()
    test_image_path = os.path.join(cf.test_dataset_path, cf.test_folder_names[0])
    test_set = Data_loader(cf, test_image_path, cf.test_samples, cf.resize_image_test)
    test_set.Load_dataset(cf.test_batch_size)
    prog_bar = ProgressBar(test_set.num_batches)
    for i in range(test_set.num_batches):
        batch_x, batch_names = test_set.Next_batch_pred(cf.test_batch_size)
        feed_dict = {sb.model.simb_image: batch_x, sb.model.simb_is_training: False}
        simbol_list = [sb.model.annotation_pred]
        sess_return = sess.run(simbol_list, feed_dict)
        pred = sess_return[0]
        save_prediction(cf.predict_output, pred, batch_names)
        prog_bar.update()
    predict_time = time.time() - predict_time
    print("\t Time: %ds" % (predict_time))

def restore_session(cf, sess, sb):
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = Model_IO()
    # Restore session
    if cf.pretrained_model:
        if cf.load_model == 'tensorflow':
            print ('Loading model ...')
            if cf.weight_only:
                saver.Load_weights(cf, sess)
            else:
                saver.Load_model(cf, sess)
        elif cf.load_model == 'keras':
            print ('Loading weights ...')
            if cf.weight_only:
                saver.Manual_weight_load(cf, sess)
            else:
                sb.model.simb_image, sb.model.logits, _  = saver.Load_keras_model(cf, sess, sb)
    return saver, sb

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
    
    sess = tf.Session()
    # training step
    if cf.train:
        #Create symbol builder with all the parameters needed (model, loss, optimizers,...)
        sb = Symbol_Builder(cf, cf.size_image_train)
        saver, sb = restore_session(cf, sess, sb)
        #merge all the previous summaries
        sb.tensorBoard.set_up() 
        print ('Starting training ...')
        Train(cf, sess, sb, saver)
    # Validation step
    if cf.validation:
        if not cf.train:
            sb = Symbol_Builder(cf, cf.size_image_valid)
            saver, sb = restore_session(cf, sess, sb)
            #merge all the previous summaries
            sb.tensorBoard.set_up() 
        print ('Starting validation ...')
        Validation(cf, sess, sb)
    # Test step
    if cf.test:
        if not cf.train and not cf.validation:
            sb = Symbol_Builder(cf, cf.size_image_test)
            saver, sb = restore_session(cf, sess, sb)
            #merge all the previous summaries
            sb.tensorBoard.set_up() 
        print ('Starting testing ...')
        if cf.predict_test:
            Predict(cf, sess, sb)
        else:
            Test(cf, sess, sb)
    total_time = time.time() - start_time    
    print(' Experiment finished: %ds ' % (total_time))

# Entry point of the script
if __name__ == "__main__":
    main()