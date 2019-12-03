import tensorflow as tf
import pickle
import misc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import feat_data_loader
from tqdm import tqdm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neural_network
import cv2


"""
Trains a model to produce a latent point corresponding to a set of features.
Trained using the latent space of the GAN.
"""

def load_generator(gan_filename='models/pg_gan/karras2018iclr-celebahq-1024x1024.pkl'):
    # Import official CelebA-HQ networks.
    with open(gan_filename, 'rb') as file:
        G, D, Gs =  pickle.load(file)
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
    G = load_generator()

    NUM_EPOCHS = 50
    ITERATIONS_PER_EPOCH = 10
    BATCHSIZE = 500

    #reg_param = {'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 2000, 'max_depth': 10, 'bootstrap': True}
    models = {#'lin_reg' : LinearRegression(),
        #'elasticnet': sklearn.linear_model.ElasticNet(),
        #'lassolars': sklearn.linear_model.LassoLars(),
        'neural_1': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64,)),
        #'neural_2': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64,32, 64)),
        #'neural_3': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(128)),
        'neural_4': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(128,128)),
        'neural_5': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(256,256)),
        'neural_6': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(512,512)),
        'neural_7': sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(512,512,512,512))}
    
    data_gen = feat_data_loader.f_z_generator(128)
    for epoch in range(NUM_EPOCHS):
        print("Loading data epoch {}...".format(epoch))
        fxz = [(feats, images, latents) for (i, (latents, images, feats)) in zip(tqdm(range(BATCHSIZE)), data_gen)]
        f = np.squeeze(np.array([x[0] for x in fxz]))
        x = np.squeeze(np.array([x[1] for x in fxz]))
        z = np.squeeze(np.array([x[2] for x in fxz]))
        for i in range(20):
            cv2.imwrite("inverter_images/output_{}.png".format(i), x[i, :])
        for name, regr in models.items():
            
                #rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
            print("Training {} epoch {}...".format(name, epoch))
            for i in tqdm(range(ITERATIONS_PER_EPOCH)):
                regr.partial_fit(f, z)
        
            #print(rf_random.best_params_)
            #regr = rf_random.best_estimator_
            pkl_filename = "models/inverter/inverter_ar2_{}_{}.pkl".format(name, BATCHSIZE*NUM_EPOCHS)
            with open(pkl_filename, 'wb') as file:
                pickle.dump(regr, file)

            for i in range(20):
                test_z = regr.predict(f[i, :].reshape(1, -1))
                labels = np.zeros([test_z.shape[0], 0], np.float32)
                test_predictions = G.run(test_z, labels, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
        
                test_predictions = np.squeeze(test_predictions)
                test_predictions = np.transpose(test_predictions, (1,2,0))
                test_predictions = cv2.cvtColor(test_predictions, cv2.COLOR_BGR2RGB)

                cv2.imwrite("inverter_images/reconstruction_{}_{}.png".format(name, i), test_predictions)
        
            #reconstruct_and_save_images(i, images, test_predictions)