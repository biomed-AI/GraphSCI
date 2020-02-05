
import tensorflow as tf
from loss import ZINB

flags = tf.flags
FLAGS = flags.FLAGS

class OptimizerSCI(object):

    def __init__(self, preds, labels, model, num_nodes, num_features, pos_weight_adj, norm_adj, global_step, ridge=0.):

        preds_adj, preds_express = preds
        labels_adj, labels_express = labels

        self.cost_adj = norm_adj * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_adj, targets=labels_adj, pos_weight=pos_weight_adj))
        # express-loss: zinb-loss
        zinb = ZINB(model.z_express_pi, theta=model.z_express_disp, ridge_lambda=ridge)
        self.cost_express = zinb.loss(tf.reshape(labels_express, [num_features, num_nodes]), 
                                      tf.reshape(preds_express, [num_features, num_nodes]))
        self.log_lik = self.cost_adj + self.cost_express
        
        # KL divergence
        self.kl_adj = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_adj_log_std - tf.square(model.z_adj_mean) - \
                                                         tf.square(tf.exp(model.z_adj_log_std)), 1))
        self.kl_express = (FLAGS.weight_decay * 0.5 / num_features) * tf.reduce_mean(tf.square(tf.subtract(tf.reshape(preds_express, [num_features, num_nodes]), tf.reshape(labels_express, [num_features, num_nodes]))))
        self.kl = self.kl_adj - self.kl_express

        self.cost = self.log_lik - self.kl

        # self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        # self.opt_op = self.optimizer.minimize(self.cost)
        # self.grads_vars = self.optimizer.compute_gradients(self.cost)

        initial_learning_rate = FLAGS.learning_rate
        self.learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step, decay_steps=50, decay_rate=0.9, staircase=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)