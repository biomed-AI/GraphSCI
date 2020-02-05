
from layers import GraphConvolution, Dense, InnerDecoder, MultiplyLayer
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

class Model(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class GraphSCI(Model):
    
    def __init__(self, placeholders, num_features, num_nodes):
        super(GraphSCI, self).__init__()

        self.input_dim = num_features
        self.num_nodes = num_nodes
        self.adj = placeholders['adj']
        self.inputs = placeholders['features']
        self.dropout = placeholders['dropout']
        self.size_factors = placeholders['size_factors']
        self.is_training = placeholders['is_training']

        self.build()

    def _build(self):

        self.adj_hidden1 = GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            adj=self.adj,
                                            act=tf.nn.tanh,
                                            dropout=self.dropout,
                                            logging=self.logging)(self.inputs)

        self.adj_hidden2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            adj=self.adj,
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging)(self.adj_hidden1)
    
        self.z_adj_mean = GraphConvolution(input_dim=FLAGS.hidden2,
                                           output_dim=self.num_nodes,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.adj_hidden2)

        self.z_adj_log_std = GraphConvolution(input_dim=FLAGS.hidden2,
                                              output_dim=self.num_nodes,
                                              adj=self.adj,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.adj_hidden2)

        self.z_adj = self.z_adj_mean + tf.random_normal([self.num_nodes, self.num_nodes]) * tf.exp(self.z_adj_log_std)

        self.express_inner = MultiplyLayer(num_nodes=self.num_nodes,
                                           act=tf.nn.relu,
                                           dropout=self.dropout,
                                           logging=self.logging)((tf.transpose(self.inputs), self.z_adj))

        self.express_hidden1 = Dense(input_dim=self.num_nodes,
                                     output_dim=FLAGS.hidden1,
                                     act=tf.nn.relu,
                                     is_training=self.is_training,
                                     dropout=self.dropout,
                                     logging=self.logging)(self.express_inner)
        
        self.express_hidden2 = Dense(input_dim=FLAGS.hidden1,
                                     output_dim=FLAGS.hidden2,
                                     act=tf.nn.relu,
                                     is_training=self.is_training,
                                     dropout=self.dropout,
                                     logging=self.logging)(self.express_hidden1)

        # self.z_express_mean_KL = Dense(input_dim=FLAGS.hidden2,
        #                                output_dim=self.num_nodes,
        #                                act=lambda x: x,
        #                                dropout=self.dropout)(self.express_hidden2)

        # self.z_express_log_std_KL = Dense(input_dim=FLAGS.hidden2,
        #                                   output_dim=self.num_nodes,
        #                                   act=lambda x: x,
        #                                   dropout=self.dropout)(self.express_hidden2)

        # self.z_express_norm = (self.z_express_mean_KL + tf.random_normal([self.input_dim, self.num_nodes]) * tf.exp(self.z_express_log_std_KL))


        # self.z_adj, self.z_express_inner = InnerDecoder(input_dim=self.num_nodes,
        #                                                 dropout=self.dropout,
        #                                                 act=lambda x: x,
        #                                                 logging=self.logging)((self.z_adj_norm, self.z_express_norm))

        
        self.z_express_pi = Dense(input_dim=FLAGS.hidden2,
                                  output_dim=self.num_nodes,
                                  act=tf.nn.sigmoid,
                                  dropout=self.dropout,
                                  logging=self.logging)(self.express_hidden2)

        self.z_express_disp = Dense(input_dim=FLAGS.hidden2,
                                    output_dim=self.num_nodes,
                                    act=lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4),
                                    dropout=self.dropout,
                                    logging=self.logging)(self.express_hidden2)

        self.z_express_mean = Dense(input_dim=FLAGS.hidden2,
                                    output_dim=self.num_nodes,
                                    act=lambda x: tf.clip_by_value(tf.exp(x), 1e-5, 1e6),
                                    dropout=self.dropout,
                                    logging=self.logging)(self.express_hidden2)

        self.z_express = self.z_express_mean * tf.reshape(self.size_factors, (-1,1))
        