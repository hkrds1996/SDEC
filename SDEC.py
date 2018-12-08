"""
Implementation for Semi-supervised deep embedded clustering as described in paper:
        Ren Y, Hu K, Dai X, et al. Semi-supervised deep embedded clustering[J]. Neurocomputing, 2019, 325: 121-130.
Usage:
    Put the weights of Pretrained autoencoder for datasets in dir "./ae_weights/datasets_ae_weights/datasets_ae_weights.h5"
        python SDEC.py datasets
Author:
    Kangrong Hu. 2018.12.08
    revised from the version of IDEC.py Xifeng Guo. 2017.1.30
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from time import time
import numpy as np
import keras
from collections import Counter
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from sklearn import metrics

def convert2int(x):
    return np.int(x)

def vex2int(x):
    C2=np.vectorize(convert2int)
    return C2(x)

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)

    # output
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)

def MYloss(y_in,y_out):
    return K.sum(y_out,axis=-1)

class SConcatenate(Layer):

    def __init__(self, beta=1,**kwargs):
        super(SConcatenate, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta=beta

    def call(self, inputs):
        return inputs[1]
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(SConcatenate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0,**kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SDEC(object):
    def __init__(self,
                 dims,
                 x,
                 N,
                 n_clusters=10,
                 alpha=1.0,
                 gamma=0.16,
                 beta=0.1,
                 batch_size=256,
                 laster_batch_size=10):

        super(SDEC, self).__init__()
        self.dims = dims
        self.x=x
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.beta=beta
        self.N=N
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.laster_batch_size=laster_batch_size
        self.autoencoder = autoencoder(self.dims)

    def initialize_model(self, optimizer, epochs=200,ae_weights=None):
        if ae_weights is not None:  # load pretrained weights of autoencoder
            self.autoencoder.load_weights(ae_weights)
        else:
            self.autoencoder.compile(loss='mse', optimizer=optimizer) 
            self.autoencoder.fit(self.x, self.x, batch_size=self.batch_size, epochs=epochs)
            self.autoencoder.save_weights(ae_weights)
        self.optimizer=optimizer
        self.hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=self.hidden)

        # prepare SDEC model
        self.clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.hidden)
        self.F_input=Input(shape=(self.N,),name="F")
        self.F_input_laster=Input(shape=(self.laster_batch_size,),name="F_input_laster")
        self.Loss=SConcatenate(name="Loss_layer")([self.hidden,self.F_input])
        self.Loss2=SConcatenate(name="Loss_layer")([self.hidden,self.F_input_laster])
        self.model = Model(inputs=[self.autoencoder.input,self.F_input],outputs=[self.clustering_layer,self.Loss])
        self.model.compile(loss={"clustering":"kld","Loss_layer":MYloss},loss_weights=[1, self.gamma],optimizer=self.optimizer)

    def load_weights(self, weights_path):  # load weights of SDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        encoder = Model(self.model.input[0], self.model.get_layer('encoder_%d' % (self.n_stacks - 1)).output)
        return encoder.predict(x)

    def predict_clusters(self, x,verbose=0):  # predict cluster labels using the output of clustering layer
        Predict=Model(inputs=self.model.input[0],outputs=self.model.get_layer(name='clustering').output)
        q = Predict.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   save_dir='./results/sdec'):

        print 'Update interval', update_interval
        save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
        print 'Save interval', save_interval
        
        #building F batch
        import csv, os
        if not os.path.exists(save_dir+"F.npy"):
            constraint=int(self.beta*y.shape[0]);
            F_index=np.zeros((2*constraint,3));
            for i in range(constraint):
                index1,index2=np.random.randint(0,y.shape[0],2);
                if (y[index1]==y[index2]):
                    F_index[i*2,0]=int(index1);
                    F_index[i*2,1]=int(index2);
                    F_index[i*2,2]=1;
                    F_index[i*2+1,0]=int(index2);
                    F_index[i*2+1,1]=int(index1);
                    F_index[i*2+1,2]=1;
                else:
                    F_index[i*2,0]=int(index1);
                    F_index[i*2,1]=int(index2);
                    F_index[i*2,2]=-1;
                    F_index[i*2+1,0]=int(index2);
                    F_index[i*2+1,1]=int(index1);
                    F_index[i*2+1,2]=-1;
            a1 = F_index[:,::-1].T  
            a2 = np.lexsort(a1)  
            F = F_index[a2]  
            np.save(save_dir+"F.npy",F)
        else:
            fileF=np.load(save_dir+"F.npy")
            F=fileF
        F.astype(int)
        # initialize cluster centers using k-means
        print 'Initializing cluster centers with k-means.'
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # logging file
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logfile = file(save_dir + '/sdec_log.csv', 'wb')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'Q','L','Lc','Lf'])
        logwriter.writeheader()
        loss = [0,0,0]
        index = 0
        Flase_y=np.zeros((self.batch_size,10))
        encoder = Model(self.model.input[0], self.model.get_layer('encoder_%d' % (self.n_stacks - 1)).output)
        feature=encoder.predict(x)
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                Predict=Model(inputs=self.model.input[0],outputs=self.model.get_layer(name='clustering').output)
                q = Predict.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                encoder = Model(self.model.input[0], self.model.get_layer('encoder_%d' % (self.n_stacks - 1)).output)
                feature=encoder.predict(x)
                if y is not None:
                    acc = np.round(cluster_acc(y, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    Q=np.mean(q)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, Q=Q,L=loss[0],Lc=loss[1],Lf=loss[2])
                    logwriter.writerow(logdict)
                    print 'Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, 'Q',Q,'; loss=', loss
                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print 'delta_label ', delta_label, '< tol ', tol
                    print 'Reached tolerance threshold. Stopping training.'
                    logfile.close()
                    break
            keras.callbacks.BaseLogger()
            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:
                # updata the data of loss
                A=F[:,0];
                F.astype(int)
                F_index=np.where((A>=index * self.batch_size));
                F_batch=np.zeros((self.laster_batch_size,x.shape[0]));
                D_feature=np.zeros((self.laster_batch_size,x.shape[0]));
                for indexF in F_index[0]:
                    F_batch[int(F[int(indexF),0]-index * self.batch_size),int(F[int(indexF),1])]=F[int(indexF),2];
                    D_feature[int(F[int(indexF),0]-index * self.batch_size),int(F[int(indexF),1])]=np.sum((feature[int(F[int(indexF),0])]-feature[int(F[int(indexF),1])])**2);
                N1=np.sum(F_batch==1);
                F_1=np.abs(F_batch)+F_batch;
                F_2=np.abs(np.abs(F_batch)-F_batch);
                N1=np.sum(np.sum(F_1));
                N2=np.sum(np.sum(F_2));
                F1=F_1/N1+np.abs(F_2);
                F2=F_2/N2+np.abs(F_1);
                inputFF=F_batch*D_feature*F1*F2
                loss = self.model.train_on_batch(x=[x[index * self.batch_size::],inputFF],
                                                 y=[p[index * self.batch_size::],np.zeros((inputFF.shape[0],10))])
                index = 0           
            else:
                # updata the data of loss
                M=feature;
                A=F[:,0];
                F_index=np.where((A<(index + 1) * self.batch_size)&(A>=index * self.batch_size));
                F_batch=np.zeros((self.batch_size,x.shape[0]));
                D_feature=np.zeros((self.batch_size,x.shape[0]));
                for indexF in F_index[0]:
                    F_batch[int(F[int(indexF),0]-index * self.batch_size),int(F[int(indexF),1])]=F[int(indexF),2];
                    D_feature[int(F[int(indexF),0]-index * self.batch_size),int(F[int(indexF),1])]=np.sum((feature[int(F[int(indexF),0])]-feature[int(F[int(indexF),1])])**2);
                N1=np.sum(F_batch==1);
                F_1=np.abs(F_batch)+F_batch;
                F_2=np.abs(np.abs(F_batch)-F_batch);
                N1=np.sum(np.sum(F_1));
                N2=np.sum(np.sum(F_2));
                F1=F_1/N1+np.abs(F_2);
                F2=F_2/N2+np.abs(F_1);
                inputFF=F_batch*D_feature*F1*F2
                loss = self.model.train_on_batch(x=[x[index * self.batch_size:(index + 1) * self.batch_size],inputFF],y=[p[index * self.batch_size:(index + 1) * self.batch_size],Flase_y])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save SDEC model checkpoints
                print 'saving model to:', save_dir + '/SDEC_model_' + str(ite) + '.h5'
                self.model.save_weights(save_dir + '/SDEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print 'saving model to:', save_dir + '/SDEC_model_final.h5'
        self.model.save_weights(save_dir + '/SDEC_model_final.h5')
        return y_pred
    
def sdec(dataset="mnist",gamma=0.1,beta=1,maxiter=2e4,update_interval=20,tol=0.00001,batch_size=256):   
    """arguements:
    dataset:choice the datasets that you want to run
    gamma: The Lambda in the lecture
    beta: the proportion of information we have known about the sample
    """
    maxiter=maxiter;
    gamma=gamma;
    update_interval=update_interval;
    tol=tol;
    beta=beta;
    batch_size=batch_size;
    ae_weights=("ae_weights/"+dataset+"_ae_weights/"+dataset+"_ae_weights.h5")
    
    # load dataset
    from datasets import load_mnist, load_usps,load_stl,load_cifar
    if dataset == 'mnist':  # recommends: n_clusters=10, update_interval=140
        x, y = load_mnist('./data/mnist/mnist.npz')
        update_interval=140
    elif dataset == 'usps':  # recommends: n_clusters=10, update_interval=30
        x, y = load_usps('data/usps')
        update_interval=30
    elif dataset=="stl":
        import numpy as np
        x,y=load_stl()
        update_interval=20
    elif dataset=="cifar_10":
        x,y=load_cifar()
        update_interval=40
    beta=beta
    print gamma,dataset,beta
    # prepare the SDEC model
    try:
        count = Counter(y)
    except:
        count = Counter(y[:,0])
    n_clusters=len(count)
    save_dir='results/sdec_dataset:'+dataset+" gamma:"+str(gamma)
    laster_batch_size=x.shape[0]%batch_size
    dec = SDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, N=x.shape[0],x=x,batch_size=batch_size,laster_batch_size=laster_batch_size,gamma=gamma,beta=beta)
    dec.initialize_model(optimizer=SGD(lr=0.01, momentum=0.9),
                             ae_weights=ae_weights)
    dec.model.summary()
    t0 = time()
    y_pred = dec.clustering(x, y=y, tol=tol, maxiter=maxiter,
                                update_interval=update_interval, save_dir=save_dir)
    plot_model(dec.model, to_file='sdecmodel.png', show_shapes=True)
    print 'acc:', cluster_acc(y, y_pred)
    print 'clustering time: ', (time() - t0)

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist', choices=['mnist', 'usps', 'cifar_10','stl'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--gamma', default=10**-5, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--beta', default=0.1, type=int)
    args = parser.parse_args()
    print args
    sdec(args.dataset,args.gamma,args.beta,args.maxiter,args.update_interval,args.tol,args.batch_size);    
    
