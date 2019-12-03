import tensorflow as tf
import cv2

import pickle
import numpy as np

from head_pose.mark_detector import MarkDetector
from queue import Queue
from threading import Thread

print("OpenCV version: {}".format(cv2.__version__))
# multiprocessing may not work on Windows and macOS, check OS for safety.
#detect_os()

def load_models(gan_filename='models/pg_gan/karras2018iclr-celebahq-1024x1024.pkl', inverter_filename='models/inverter/inverter_randforest_7000.pkl'):
    # Import official CelebA-HQ networks.
    with open(gan_filename, 'rb') as file:
        G, D, Gs = pickle.load(file)
        # G = Instantaneous snapshot of the generator, mainly useful for resuming a previous training run.
        # D = Instantaneous snapshot of the discriminator, mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator, yielding higher-quality results than the instantaneous snapshot.
    #F = tf.keras.models.load_model(feature_extractor_filename)
    with open(inverter_filename, 'rb') as file:
        I = pickle.load(file)
    return Gs, I

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
def calculate_facial_features(frame, facebox, CNN_INPUT_SIZE, mark_detector,  age_net, gender_net):
    # Detect landmarks from image of 128x128.
    face_img_og = frame[facebox[1]: facebox[3],
                     facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img_og, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    marks = mark_detector.detect_marks([face_img])
    age_gender_face = cv2.dnn.blobFromImage(face_img_og, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
    gender_net.setInput(age_gender_face)
    gender_preds = gender_net.forward()
    gender = gender_preds[0].argmax()
    #print("Gender : " + gender_list[gender])
    #Predict Age
    age_net.setInput(age_gender_face)
    age_preds = age_net.forward()
    age = age_preds[0].argmax()

    # Convert the marks locations from local CNN to global image.
    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]

    # Write text on image
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.25
    fontColor              = (0,255,0)
    lineType               = 2

    display_vars = {'age': age_list[age], 'gender': gender_list[gender], 'face_pos_0': marks[0, 0]}
    yloc = 50
    for var_name, value in display_vars.items():
        cv2.putText(frame,'${} = {}'.format(var_name, value),
            (10, yloc),
            font,
            fontScale,
            fontColor,
            lineType)
    yloc += 50

    # Uncomment following line to show raw marks.
    mark_detector.draw_marks(
         frame, marks, color=(0, 255, 0))

    # Uncomment following line to show facebox.
    mark_detector.draw_box(frame, [facebox])

    return np.append(marks.flatten(), np.array([age, gender])).reshape(1, -1)

def process_video_capture(inverter_filename='models/inverter/inverter_randforest_7000.pkl'):
    CNN_INPUT_SIZE = 128
    sess = tf.InteractiveSession()
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    #tm = cv2.TickMeter()

    G, I = load_models(inverter_filename=inverter_filename)
    age_net = cv2.dnn.readNetFromCaffe('models/race_age/deploy_age.prototxt', 'models/race_age/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('models/race_age/deploy_gender.prototxt', 'models/race_age/gender_net.caffemodel')

    """

    """
    feat_queue = Queue(maxsize=5)

    def gen_feats_task():
        while True:
            frame_got, frame = video_capture.read()
            if frame_got is False:
                break
            # Crop it if frame is larger than expected.
            #frame = frame[0:480, 300:940]
            # If frame comes from webcam, flip it so it looks like a mirror.
            frame = cv2.flip(frame, 2)

            # Feed frame to image queue.
            facebox = mark_detector.extract_cnn_facebox(frame)

            if facebox is not None:
                feats = calculate_facial_features(frame, facebox, CNN_INPUT_SIZE, mark_detector, age_net, gender_net)
                feat_queue.put(feats)

    gen_feats_thread = Thread(target=gen_feats_task, daemon=True)
    gen_feats_thread.start()

    i = 0
    #intercept = np.zeros(Z_DIM)
    while True:
        #intercept += np.random.normal(scale=0.1, size=intercept.shape)
        #if np.dot(intercept, intercept) > 1:
        #    intercept *= 0.5
        # Read frame, crop it, flip it, suits your needs.
        feats = feat_queue.get()
        viewer_latent = I.predict(feats)# + intercept
        labels = np.zeros([viewer_latent.shape[0], 0], np.float32)
        viewer_generated = G.run(viewer_latent, labels,out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
        viewer_generated = np.squeeze(viewer_generated)
        viewer_generated = np.transpose(viewer_generated, (1,2,0))
        viewer_generated = cv2.cvtColor(viewer_generated, cv2.COLOR_BGR2RGB)
        #facebox = mark_detector.extract_cnn_facebox(viewer_generated)
        #if facebox is not None:
        #    feats_gen = calculate_facial_features(viewer_generated, facebox, CNN_INPUT_SIZE, mark_detector, age_net, gender_net)
        #    print("Frame: {}, mean diff in features: {} ".format(i, np.mean(np.abs(feats_gen - feats))))
        # Show preview.

        #cv2.imwrite("images/output_{}.png".format(i), viewer_generated)

        cv2.imshow("Projection", viewer_generated)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
        i += 1
    video_capture.release()


if __name__ == '__main__':
    inv = 'models/inverter/inverter_ar_neural_5_10000.pkl'
    process_video_capture(inverter_filename=inv)
