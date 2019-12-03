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
from sklearn.model_selection import RandomizedSearchCV
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neural_network
import pickle
import cv2
from joblib import dump


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
    F_DIM = 138
    G = load_generator()

    NUM_DATAPOINT = 10000
    print("Loading data...")
    fxz = [(feats, images, latents) for (i, (latents, images, feats)) in zip(tqdm(range(NUM_DATAPOINT)), feat_data_loader.f_z_generator(128))]
    f = np.squeeze(np.array([x[0] for x in fxz]))
    x = np.squeeze(np.array([x[1] for x in fxz]))
    z = np.squeeze(np.array([x[2] for x in fxz]))


    #reg_param = {'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 2000, 'max_depth': 10, 'bootstrap': True}
    models = {#'lin_reg' : LinearRegression(),
        #'elasticnet': sklearn.linear_model.ElasticNet(),
        #'lassolars': sklearn.linear_model.LassoLars(),
        #'neural_1': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64,)),
        #'neural_2': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64,32, 64)),
        #'neural_3': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(128)),
        'neural_4': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(128,128)),
        'neural_5': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(256,256)),
        'neural_6': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(512,512,512))}

    for name, regr in models.items():
        #rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        print("Training...")
        regr.fit(f, z)

        #print(rf_random.best_params_)
        #regr = rf_random.best_estimator_


        for i in range(20):
            test_z = regr.predict(f[i, :].reshape(1, -1))
            labels = np.zeros([test_z.shape[0], 0], np.float32)
            test_predictions = G.run(test_z, labels, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)

            images = x[i, :]
            test_predictions = np.squeeze(test_predictions)
            test_predictions = np.transpose(test_predictions, (1,2,0))
            test_predictions = cv2.cvtColor(test_predictions, cv2.COLOR_BGR2RGB)
            cv2.imwrite("inverter_images/output_{}.png".format(i), images)
            cv2.imwrite("inverter_images/reconstruction_{}_{}.png".format(name, i), test_predictions)

            #reconstruct_and_save_images(i, images, test_predictions)
        pkl_filename = "models/inverter/inverter_ar_{}_{}.pkl".format(name, NUM_DATAPOINT)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(regr, file)
