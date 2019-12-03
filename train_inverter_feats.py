import tensorflow as tf
import pickle
import misc
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import feat_data_loader
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import sklearn.linear_model
import pickle
import cv2
from joblib import dump

class FeatureLoss(Regression):
    """Squared loss traditional used in linear regression."""
    cdef double loss(self, double p, double y) nogil:
        return 0.5 * (p - y) * (p - y)

    cdef double _dloss(self, double p, double y) nogil:
        return p - y

    def __reduce__(self):
        return SquaredLoss, ()

def LinearLatentModel(sklearn.linear_model.BaseSGDRegressor):
    loss_functions = {
        "feature_loss": (FeatureLoss, ),
    }
    @abstractmethod
    def __init__(self, loss="feature_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
                 shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON,
                 random_state=None, learning_rate="invscaling", eta0=0.01,
                 power_t=0.25, early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, warm_start=False, average=False):
        super().__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, warm_start=warm_start,
            average=average)

"""
Trains a model to produce a latent point corresponding to a set of features.
Trained using the latent space of the GAN.
"""

def load_generator(gan_filename='models/pg_gan/karras2018iclr-celebahq-1024x1024.pkl'):
    # Import official CelebA-HQ networks.
    with open(gan_filename, 'rb') as file:
        G, D, Gs = pickle.load(file)
    return G

def reconstruct_and_save_images(epoch, test_images, predictions):
    print("Saving reconstruction...")
    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, 2*i+1)
        predi = misc.convert_to_pil_image(test_images[i])
        plt.imshow(predi)
        plt.axis('off')
        plt.subplot(4, 4, 2*i+2)
        predi = misc.convert_to_pil_image(predictions[i])
        plt.imshow(predi)
        plt.axis('off')

    plt.savefig("inverter_images/" + 'reconstruction_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    plt.clf()

# Start training
with tf.Session() as sess:
    Z_DIM = 512
    F_DIM = 136
    G = load_generator()

    NUM_DATAPOINT = 5000
    print("Loading data...")
    fxz = [(feats.reshape(-1, F_DIM), images, latents) for (i, (latents, images, feats)) in zip(tqdm(range(NUM_DATAPOINT)), feat_data_loader.f_z_generator(128))]
    f = np.squeeze(np.array([x[0] for x in fxz]))
    x = np.squeeze(np.array([x[1] for x in fxz]))
    z = np.squeeze(np.array([x[2] for x in fxz]))

    regr = LinearLatentModel()
    print("Training...")
    regr.fit(f, z)


    for i in range(20):
        test_z = regr.predict(f[i, :].reshape(1, -1))
        labels = np.zeros([test_z.shape[0], 0], np.float32)
        test_predictions = G.run(test_z, labels, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)

        images = x[i, :]
        test_predictions = np.squeeze(test_predictions)
        test_predictions = np.transpose(test_predictions, (1,2,0))
        test_predictions = cv2.cvtColor(test_predictions, cv2.COLOR_BGR2RGB)
        cv2.imwrite("inverter_images/output_{}.png".format(i), images)
        cv2.imwrite("inverter_images/reconstruction_{}.png".format(i), test_predictions)

        #reconstruct_and_save_images(i, images, test_predictions)
    pkl_filename = "models/inverter/inverter.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(regr, file)
