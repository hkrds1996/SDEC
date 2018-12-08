import numpy as np
'''
from datasets import load_mnist, load_stl, load_usps,load_cifar. 
'''

def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        for did in did_to_cat.keys():
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did_to_cat.has_key(did):
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000]
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], x.size / x.shape[0]))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})

def load_mnist_data(data_path='./data/mnist/mnist.npz'):
    f = np.load(data_path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def load_mnist(data_path='./data/mnist/mnist.npz'):
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_mnist_data(data_path)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 50.)  # normalize as it does in DEC paper
    print('MNIST samples', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [map(float, line.split()) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [map(float, line.split()) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y



import pickle
import sys,os
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def hog_picture(hog, resolution):
    from scipy.misc import imrotate
    glyph1 = np.zeros((resolution, resolution), dtype=np.uint8)
    glyph1[:,int(round(resolution/2))-1:int(round(resolution / 2)) + 1] = 255
    glyph = np.zeros((resolution, resolution, 9), dtype=np.uint8)
    glyph[:, :, 0] = glyph1
    for i in xrange(1, 9):
        glyph[:, :, i] = imrotate(glyph1, -i * 20)
    shape = hog.shape
    clamped_hog = hog.copy()
    clamped_hog[hog < 0] = 0
    image = np.zeros((resolution * shape[0], resolution * shape[1]), dtype=np.float32)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            for k in xrange(9):
                image[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution] = np.maximum(image[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution], clamped_hog[i, j, k] * glyph[:, :, k])

    return image
def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = pickle.load(f)
    else:
        d = pickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels
def load_data(data_dir='./data/cifar_10'):
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    num_train_samples = 50000
    import tarfile
    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')
    if not os.path.exists(data_dir+'/cifar-10-batches-py/test_batch'):
        if not os.path.exists(data_dir+'/cifar-10-python.tar.gz'):
            os.system('wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P %s' % data_dir)
        tar = tarfile.open(data_dir+"/cifar-10-python.tar.gz")
        tar.extractall(data_dir)
        tar.close()
    
    for i in range(1, 6):
        fpath = os.path.join(data_dir+'/cifar-10-batches-py/data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels
    fpath = os.path.join(data_dir+'/cifar-10-batches-py/test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    return (x_train, y_train), (x_test, y_test)

def make_cifar(data_dir='./data/cifar_10'):
    """Get the features of cifar_10 datasets
    #Save the features as .npy file
    """
    import cv
    import cv2
    from joblib import Parallel, delayed
    import features
    (x_train, y_train), (x_test, y_test)=load_data()
    print(x_train.shape)
    x=np.concatenate((x_train,x_test))
    x = x.transpose((0,3,2,1))
    y=np.concatenate((y_train,y_test))
    n_jobs = 10
    cmap_size = (8,8)
    N = x.shape[0]
    X=x
    H = np.asarray(Parallel(n_jobs=n_jobs)( delayed(features.hog)(X[i]) for i in xrange(N) ))
    H = H.reshape((H.shape[0], H.size/N))
    X_small = np.asarray(Parallel(n_jobs=n_jobs)( delayed(cv2.resize)(X[i], cmap_size) for i in xrange(N) ))
    crcb = np.asarray(Parallel(n_jobs=n_jobs)( delayed(cv2.cvtColor)(X_small[i], cv.CV_RGB2YCrCb) for i in xrange(N) ))
    crcb = crcb[:,:,:,1:]
    crcb = crcb.reshape((crcb.shape[0], crcb.size/N))
    feature = np.concatenate(((H-0.2)*10.0, (crcb-128.0)/10.0), axis=1)
    x=feature
    image=X[:,:,:,[2,1,0]]
    x=x.astype(np.float64)
    y=np.array(y)
    y=y
    p = np.random.permutation(X.shape[0])
    x=x[p]
    y=y[p]
    np.save(os.path.join(data_dir, 'cifar_10.npy'), {'data': x, 'label': y})

def load_cifar(data_dir='./data/cifar_10'):
    """Loads cifar_10 datasets
    #Returns
        Tuple of Numpy arrays: `(x, y)`.
    """
    import os
    if not os.path.exists(data_dir+'/cifar_10.npy'):
        print('making cifar-10 hog features')
        make_cifar(data_dir)
        print('hog features saved in ' +data_dir)
    data = np.load(os.path.join(data_dir, 'cifar_10.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], x.size / x.shape[0])).astype('float64')
    y = y.reshape((y.size,))
    print('cifar-10 samples', x.shape)
    return x, y

def load_stl_data(fname):
    """get the features of pictures
    """
    import cv,cv2
    from joblib import Parallel, delayed
    import features
    X = np.fromfile(fname, dtype=np.uint8)
    X = X.reshape((X.size/3/96/96, 3, 96, 96)).transpose((0,3,2,1))
    n_jobs = 10
    cmap_size = (8,8)
    N = X.shape[0]
    H = np.asarray(Parallel(n_jobs=n_jobs)( delayed(features.hog)(X[i]) for i in xrange(N) ))
    H_img = np.repeat(np.asarray([ hog_picture(H[i], 9) for i in xrange(100) ])[:, :,:,np.newaxis], 3, 3)
    H = H.reshape((H.shape[0], H.size/N))
    X_small = np.asarray(Parallel(n_jobs=n_jobs)( delayed(cv2.resize)(X[i], cmap_size) for i in xrange(N) ))
    crcb = np.asarray(Parallel(n_jobs=n_jobs)( delayed(cv2.cvtColor)(X_small[i], cv.CV_RGB2YCrCb) for i in xrange(N) ))
    crcb = crcb[:,:,:,1:]
    crcb = crcb.reshape((crcb.shape[0], crcb.size/N))
    feature = np.concatenate(((H-0.2)*10.0, (crcb-128.0)/10.0), axis=1)
    return feature, X[:,:,:,[2,1,0]]

def make_stl(data_dir='./data/stl'):
    """Get the features of STL datasets
    #Save the features as .npy file
    """
    np.random.seed(1234)
    if not os.path.exists(data_dir+'/stl10_binary/train_X.bin'):
        if not os.path.exists(data_dir+'/stl10_binary.tar.gz'):
            os.system('wget http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz -P %s' % data_dir)
        tar = tarfile.open(data_dir+"/stl10_binary.tar.gz")
        tar.extractall(data_dir)
        tar.close()
    X_train, img_train = load_stl_data(data_dir+'/stl10_binary/'+'train_X.bin')
    X_test, img_test = load_stl_data(data_dir+'/stl10_binary/'+'test_X.bin')
    X_unlabel, img_unlabel = load_stl_data(data_dir+'/stl10_binary/'+'unlabeled_X.bin')
    Y_train = np.fromfile(data_dir+'/stl10_binary/'+'train_y.bin', dtype=np.uint8) - 1
    Y_test = np.fromfile(data_dir+'/stl10_binary/'+'test_y.bin', dtype=np.uint8) - 1
    X_total = np.concatenate((X_train, X_test), axis=0)
    img_total = np.concatenate((img_train, img_test), axis=0)
    Y_total = np.concatenate((Y_train, Y_test))
    p = np.random.permutation(X_total.shape[0])
    X_total = X_total[p]
    img_total = img_total[p]
    Y_total = Y_total[p]
    x=X_total
    y=Y_total
    np.save(os.path.join(data_dir, 'stl.npy'), {'data': x, 'label': y})

def load_stl(data_dir='./data/stl'):
    """Loads STL datasets
    #Returns
        Tuple of Numpy arrays: `(x, y)`.
    """
    import os
    if not os.path.exists(data_dir+'/stl.npy'):
        print('making stl hog features')
        make_stl(data_dir)
        print('hog features saved in ' +data_dir)
    data = np.load(os.path.join(data_dir, 'stl.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], x.size / x.shape[0])).astype('float64')
    y = y.reshape((y.size,))
    print('stl samples', x.shape)
    return x, y