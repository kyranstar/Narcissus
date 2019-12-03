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
import pickle
import cv2
from joblib import dump
from scipy.optimize import minimize
from head_pose.mark_detector import MarkDetector


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

def feats_loss(latent_pred, feats_true, G, mark_detector, sample_weights=None):
    Z_DIM = 512
    print(latent_pred.shape)
    labels = np.zeros([latent_pred.shape[0], 0], np.float32)
    loss = 0
    print("Calculating loss...")
    for i in tqdm(range(latent_pred.shape[0])):
        z_train = latent_pred[i, :].reshape(-1, Z_DIM)
        test_predictions = G.run(z_train, labels, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
        test_predictions = np.squeeze(test_predictions)
        test_predictions = np.transpose(test_predictions, (1,2,0))
        test_predictions = cv2.cvtColor(test_predictions, cv2.COLOR_BGR2RGB)
        facebox = mark_detector.extract_cnn_facebox(test_predictions)
        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks([face_img])
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            loss += np.square(marks - feats_true)
        else:
            loss += np.ones(136) * 1000
    print("loss: {}".format(loss))
    return loss

class CustomLinearModel():



# Start training
with tf.Session() as sess:
    Z_DIM = 512
    F_DIM = 136
    G = load_generator()
    mark_detector = MarkDetector()

    NUM_DATAPOINT = 50
    print("Loading data...")
    fxz = [(feats.reshape(-1, F_DIM), images, latents) for (i, (latents, images, feats)) in zip(tqdm(range(NUM_DATAPOINT)), feat_data_loader.f_z_generator(128))]
    f = np.squeeze(np.array([x[0] for x in fxz]))
    x = np.squeeze(np.array([x[1] for x in fxz]))
    z = np.squeeze(np.array([x[2] for x in fxz]))

    # the parameter search space
    beta_init = np.zeros((1,f.shape[1]))
    l2_mape_model = CustomLinearModel(G, mark_detector,
        loss_function=feats_loss,
        beta_init=beta_init,
        X=f, Y=f, regularization=0.00012
    )
    l2_mape_model.fit()
    l2_mape_model.beta


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
    pkl_filename = "models/inverter/inverter_linear_latent_feat.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(regr, file)
