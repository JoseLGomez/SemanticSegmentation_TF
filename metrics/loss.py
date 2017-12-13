import sys
import tensorflow as tf

class LossHandler(object):
    def __init__(self,cf):
        '''
        Net loader Initialization function.
        @param config: configuration parameters.
        '''
        #self.loss_func = self.__Select_loss(cf.type)
        self.cf = cf


    def loss(self, input, labels, vars):
        return self.__semanticLoss(input, labels, vars)

    def __semanticLoss(self, input, labels, vars):
        # Remove invalid labels
        neg_mask_labels = tf.squeeze(
            tf.cast(tf.equal(tf.cast(labels, tf.float32), tf.cast(self.cf.void_class * tf.ones_like(labels),
                    tf.float32)), tf.float32),axis=3)
        mask_labels = tf.abs(neg_mask_labels - 1.0)
        aux_labels = labels * tf.expand_dims(tf.cast(mask_labels,tf.int32), -1)

        # Compute loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=input,
                                                                        labels=tf.to_int64(
                                                                            tf.squeeze(aux_labels, axis=3)))
        # Add weight decay
        wd_loss = self.__weightDecayLoss(vars)
        loss = tf.reduce_mean(loss * mask_labels) + tf.add_n(wd_loss)

        return loss


    def __weightDecayLoss(self, vars):
        wd_loss = [self.cf.weight_decay * tf.nn.l2_loss(v) for v in vars if 'weights' in v.name]
        return wd_loss