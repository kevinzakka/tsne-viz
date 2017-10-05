import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
import matplotlib.mlab as mlab
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from utils import *
from config import get_config

def main(config):

    np.random.seed(config.random_seed)

    # ensure directories are setup
    prepare_dirs(config)

    # load data
    X_train, y_train, _, _ = load_data(config.data_dir)

    # shuffle dataset
    if config.shuffle:
        p = np.random.permutation(len(X_train))
        X_train = X_train[p]
        y_train = y_train[p]

    num_classes = len(np.unique(y_train))
    labels = np.arange(num_classes)

    # restrict to a sample because slow
    mask = np.arange(config.num_samples)
    X_sample = X_train[mask].squeeze()
    y_sample = y_train[mask]

    # grab file names for saving
    file_name = name_file(config) + '.p'
    v = '_v1'
    if config.with_images == True:
        v = '_v2'
    img_name = name_file(config) + v + '.pdf'

    if config.compute_embeddings:
        print("X_sample: {}".format(X_sample.shape))
        print("y_sample: {}".format(y_sample.shape))

        # flatten images to (N, D) for feeding to t-SNE
        X_sample_flat = np.reshape(X_sample, [X_sample.shape[0], -1])

        # compute tsne embeddings
        embeddings = TSNE(n_components=config.num_dimensions, init='pca', verbose=2).fit_transform(X_sample_flat)

        # dump
        pickle.dump(embeddings, open(config.data_dir + file_name, "wb"))

    # else load
    print("Loading embedding...")
    embeddings = pickle.load(open(config.data_dir + file_name, "rb"))

    print('Plotting...')
    if config.num_dimensions == 3:

        # safeguard
        if config.with_images == True:
            sys.exit("Cannot plot images with 3D plots.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = cm.Spectral(np.linspace(0, 1, num_classes))

        xx = embeddings[:, 0]
        yy = embeddings[:, 1]
        zz = embeddings[:, 2]

        # plot the 3D data points
        for i in range(num_classes):
            ax.scatter(xx[y_sample==i], yy[y_sample==i], zz[y_sample==i], color=colors[i], label=labels[i], s=10)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.zaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=5)
        plt.savefig(config.plot_dir + img_name, format='pdf', dpi=600)
        plt.show()

    # 2D plot
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = cm.Spectral(np.linspace(0, 1, num_classes))

        xx = embeddings[:, 0]
        yy = embeddings[:, 1]

        # plot the images
        if config.with_images == True:
            for i, (x, y) in enumerate(zip(xx, yy)):
                im = OffsetImage(X_sample[i], zoom=0.1, cmap='gray')
                ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                ax.add_artist(ab)
            ax.update_datalim(np.column_stack([xx, yy]))
            ax.autoscale()

        # plot the 2D data points
        for i in range(num_classes):
            ax.scatter(xx[y_sample==i], yy[y_sample==i], color=colors[i], label=labels[i], s=10)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=5)
        plt.savefig(config.plot_dir + img_name, format='pdf', dpi=600)
        plt.show()

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)