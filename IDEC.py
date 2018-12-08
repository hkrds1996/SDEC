"""
Re-implementation for Improved Deep Embedded Clustering as described in paper:
        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure
        Preservation. IJCAI 2017.
Usage:
    Put the weights of Pretrained autoencoder for datasets in dir "./ae_weights/datasets_ae_weights/datasets_ae_weights.h5"
        python IDEC.py datasets 
Author:
    Kangrong Hu. 2018.12.08
    revised from the version of Xifeng Guo. 2017.1.30
"""


from time import time
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from collections import Counter

from sklearn.cluster import KMeans
from sklearn import metrics

from DEC import cluster_acc, ClusteringLayer, autoencoder


class IDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):

        super(IDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dims)

    def initialize_model(self, ae_weights=None, gamma=0.1, optimizer='adam'):
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print 'Pretrained AE weights are loaded successfully.'
        else:
            print 'ae_weights must be given. E.g.'
            print '    python IDEC.py mnist --ae_weights weights.h5'
            exit()

        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        # prepare IDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])
        self.model.compile(loss={'clustering': 'kld', 'decoder_0': 'mse'},
                           loss_weights=[gamma, 1],
                           optimizer=optimizer)

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        encoder = Model(self.model.input, self.model.get_layer('encoder_%d' % (self.n_stacks - 1)).output)
        return encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   save_dir='./results/idec'):

        print 'Update interval', update_interval
        save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
        print 'Save interval', save_interval

        # initialize cluster centers using k-means
        print 'Initializing cluster centers with k-means.'
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = file(save_dir + '/idec_log.csv', 'wb')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    acc = np.round(cluster_acc(y, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print 'Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print 'delta_label ', delta_label, '< tol ', tol
                    print 'Reached tolerance threshold. Stopping training.'
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=[p[index * self.batch_size::], x[index * self.batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=[p[index * self.batch_size:(index + 1) * self.batch_size],
                                                    x[index * self.batch_size:(index + 1) * self.batch_size]])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                print 'saving model to:', save_dir + '/IDEC_model_' + str(ite) + '.h5'
                self.model.save_weights(save_dir + '/IDEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print 'saving model to:', save_dir + '/IDEC_model_final.h5'
        self.model.save_weights(save_dir + '/IDEC_model_final.h5')
        
        return y_pred
def idec(dataset="mnist",gamma=0.1,maxiter=2e4,update_interval=20,tol=0.00001,batch_size=256):    
    maxiter=maxiter;
    gamma=gamma;
    update_interval=update_interval;
    tol=tol;
    batch_size=batch_size;
    ae_weights=("ae_weights/"+dataset+"_ae_weights/"+dataset+"_ae_weights.h5")
    
    optimizer = SGD(lr=0.01, momentum=0.9)
    from datasets import load_mnist,load_usps,load_stl,load_cifar
    if dataset == 'mnist':  # recommends: n_clusters=10, update_interval=140
        x, y = load_mnist('./data/mnist/mnist.npz')
        update_interval=140
    elif dataset == 'usps':  # recommends: n_clusters=10, update_interval=30
        x, y = load_usps('data/usps')
        update_interval=30
    # prepare the IDEC model
    elif dataset=="stl":
        import numpy as np
        x,y=load_stl()
        update_interval=20
    elif dataset=="cifar_10":
        x,y=load_cifar()
        update_interval=140
    batch_size=120  
    print gamma,dataset
    try:
        count = Counter(y)
    except:
        count = Counter(y[:,0])
    n_clusters=len(count)
    save_dir='results/idec_dataset:'+dataset+" gamma:"+str(gamma)
    idec = IDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, batch_size=batch_size)
    idec.initialize_model(ae_weights=ae_weights, gamma=gamma, optimizer=optimizer)
    plot_model(idec.model, to_file='idec_model.png', show_shapes=True)
    idec.model.summary()

    # begin clustering, time not include pretraining part.
    t0 = time()
    y_pred = idec.clustering(x, y=y, tol=tol, maxiter=maxiter,
                             update_interval=update_interval, save_dir=save_dir)
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
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    print args
    idec(args.dataset,args.gamma,args.maxiter,args.update_interval,args.tol,args.batch_size);