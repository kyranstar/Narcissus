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
from head_pose.mark_detector import MarkDetector
import cv2

"""
Trains a model to produce a latent point corresponding to a set of features.
Trained using the latent space of the GAN.
"""

def load_generator(gan_filename='models/pg_gan/karras2018iclr-celebahq-1024x1024.pkl'):
    # Import official CelebA-HQ networks.
    with open(gan_filename, 'rb') as file:
        G, D, Gs = pickle.load(file)
    return G

def create_inverter(x, f_dim, z_dim):
    with tf.variable_scope('Inverter', reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x, f_dim)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dense(x, 256)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dense(x, z_dim)
        x = tf.nn.tanh(x)
    return x

# Start training
with tf.Session() as sess:
    Z_DIM = 512
    F_DIM = 136
    G = load_generator()

    f_input = tf.placeholder(tf.float32, shape=[None, F_DIM])
    z_output = create_inverter(f_input, f_dim=F_DIM, z_dim=Z_DIM)
    labels_in = tf.placeholder(tf.float32, shape=[None, 0])
    x_reconstructed = G.get_output_for(z_output, labels_in)

    #mark_detector = MarkDetector()
    #x_reconstructed = tf.squeeze(x_reconstructed)
    #x_reconstructed = tf.transpose(x_reconstructed, (1, 2, 0))

    #facebox = mark_detector.extract_cnn_facebox(x_reconstructed)
    # Calculate facial features
    #f_reconstructed = feat_data_loader.calculate_facial_features(x_reconstructed, facebox, 128, mark_detector)

    z_input = tf.placeholder(tf.float32, shape=[None, Z_DIM])
    inv_loss =  tf.square(z_output - z_input)#tf.nn.sigmoid_cross_entropy_with_logits(labels = x_constructed, logits = x_reconstructed)#inverter_loss(noise_input, inverter_output)#, real_image, rec_image)
    mean_inv_loss = tf.reduce_mean(inv_loss)

    optimizer_inv = tf.train.AdamOptimizer(learning_rate=0.001)
    inv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Inverter')
    train_inv = optimizer_inv.minimize(inv_loss, var_list=inv_vars)

    init = tf.global_variables_initializer()

    NUM_STEPS = 80000
    saver = tf.train.Saver()

    # Run the initializer
    sess.run(init)
    for i, (latents, images, feats) in enumerate( feat_data_loader.f_z_generator(128)):
        # Training
        feats = feats.reshape(-1, F_DIM)
        feed_dict = {z_input: latents, f_input: feats}
        _, il = sess.run([train_inv, mean_inv_loss],
                                feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1 or i == NUM_STEPS:
            print('Step %i: Inverter Loss: %f' % (i, il))
            saver.save(sess, 'models/inverter/inverter', global_step=i)
            test_labels = np.zeros([latents.shape[0], 0], np.float32)
            test_predictions = sess.run(x_reconstructed, feed_dict={f_input: feats, labels_in:test_labels})
            test_predictions = np.squeeze(test_predictions)
            test_predictions = np.transpose(test_predictions, (1,2,0))
            test_predictions = cv2.cvtColor(test_predictions, cv2.COLOR_BGR2RGB)
            cv2.imwrite("inverter_images/output_{}.png".format(i), images)
            cv2.imwrite("inverter_images/reconstruction_{}.png".format(i), test_predictions)
