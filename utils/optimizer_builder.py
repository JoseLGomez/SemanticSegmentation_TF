import os
import tensorflow as tf

class Optimizer_builder():
    def __init__(self):
        pass
        
    def build(self, cf):
        if cf.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=cf.learning_rate, beta1=0.5, beta2=0.999)
        elif cf.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=cf.learning_rate)
        elif cf.optimizer == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=cf.learning_rate)
        else:
            sys.exit('Optmizer model not defined: ' + cf.type)
    
    '''def update_lr(self,step):
        orig_lr = lambda: self.params.lr
        updated_lr = lambda: (tf.multiply((step - self.params.lr_decay_start), self.params.lr) / (step - self.params.lr_decay_start)) ** 0.9
        lr = tf.cond(step>self.params.lr_decay_start,updated_lr,orig_lr)
        self.__opt = self.__select_optimizer(lr)'''
        
    def minimize(self,optimizer,loss,vars):
        #self.update_lr(step)
        grads_and_vars = optimizer.compute_gradients(loss, vars)
        op_minimizer = optimizer.apply_gradients(grads_and_vars)

        return grads_and_vars, op_minimizer