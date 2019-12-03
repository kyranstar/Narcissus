import tensorflow as tf
import cv2
from argparse import ArgumentParser

import pickle
from joblib import load
import numpy as np

from head_pose.mark_detector import MarkDetector
from head_pose.os_detector import detect_os
from head_pose.pose_estimator import PoseEstimator
from head_pose.stabilizer import Stabilizer
import joblib
import frames_to_video

print("OpenCV version: {}".format(cv2.__version__))
# multiprocessing may not work on Windows and macOS, check OS for safety.
#detect_os()

def load_models(gan_filename='models/pg_gan/karras2018iclr-celebahq-1024x1024.pkl', inverter_filenames=['models/inverter/inverter_neural_3_8000_good.pkl', 'models/inverter/inverter_neural_1_8000_good.pkl']):
    # Import official CelebA-HQ networks.
    with open(gan_filename, 'rb') as file:
        G, D, Gs = pickle.load(file)
        # G = Instantaneous snapshot of the generator, mainly useful for resuming a previous training run.
        # D = Instantaneous snapshot of the discriminator, mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator, yielding higher-quality results than the instantaneous snapshot.
    #F = tf.keras.models.load_model(feature_extractor_filename)
    I = []
    for inv_fn in inverter_filenames:
        with open(inv_fn, 'rb') as file:
            I.append(pickle.load(file))
    return Gs, I

def convert_array_to_image(array):
    array = tf.reshape(array, [128, 128, 3])
    """Converts a numpy array to a PIL Image and undoes any rescaling."""
    img = PIL.Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
    return img

def calculate_facial_features(frame, facebox, tm, CNN_INPUT_SIZE, mark_detector):
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


    # Uncomment following line to show raw marks.
    mark_detector.draw_marks(
         frame, marks, color=(0, 255, 0))

    # Uncomment following line to show facebox.
    mark_detector.draw_box(frame, [facebox])
    """
    # Try pose estimation with 68 points.
    pose = pose_estimator.solve_pose_by_68_points(marks)

    # Stabilize the pose.
    steady_pose = []
    pose_np = np.array(pose).flatten()
    for value, ps_stb in zip(pose_np, pose_stabilizers):
        ps_stb.update([value])
        steady_pose.append(ps_stb.state[0])
    steady_pose = np.reshape(steady_pose, (-1, 3))
    """
    #if not frame is None:
    #     Uncomment following line to draw pose annotation on frame.
    #     pose_estimator.draw_annotation_box(
    #         frame, pose[0], pose[1], color=(255, 128, 128))

    # if not frame is None:
    #   pose_estimator.draw_annotation_box(
    #       frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

    # Uncomment following line to draw head axes on frame.
    # pose_estimator.draw_axes(frame, stabile_pose[0], stabile_pose[1])

    return marks

def process_video_capture(video_capture, save_file, inverter_filenames='models/inverter/inverter_randforest_7000.pkl'):
    Z_DIM = 512
    CNN_INPUT_SIZE = 128
    sess = tf.InteractiveSession()

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    _, sample_frame = video_capture.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()


    G, I = load_models(inverter_filenames=inverter_filenames)
    #print(I.coef_)
    #print(I.intercept_)

    i = 0
    last_image =  np.zeros((1024,1024,3))
    while True:
        #I.intercept_ += np.random.normal(scale=0.1, size=I.intercept_.shape).astype(np.float32)
        # Read frame, crop it, flip it, suits your needs.
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
            feats = calculate_facial_features(frame, facebox, tm, CNN_INPUT_SIZE, mark_detector)
            viewer1_latent = I[0].predict(feats.reshape(1, -1))
            viewer2_latent = I[1].predict(feats.reshape(1, -1))
            viewer_latent = (viewer2_latent + viewer1_latent) / 2.0
            labels = np.zeros([viewer_latent.shape[0], 0], np.float32)
            viewer_generated = G.run(viewer_latent, labels,out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
            viewer_generated = np.squeeze(viewer_generated)
            viewer_generated = np.transpose(viewer_generated, (1,2,0))
            viewer_generated = cv2.cvtColor(viewer_generated, cv2.COLOR_BGR2RGB)
            facebox = mark_detector.extract_cnn_facebox(viewer_generated)
            if facebox is not None:
                feats_gen = calculate_facial_features(viewer_generated, facebox, tm, CNN_INPUT_SIZE, mark_detector)
                print("Frame: {}, mean diff in features: {} ".format(i, np.mean(np.abs(feats_gen - feats))))
            # Show preview.
        else:
            print("Frame: {}, didn't generate face".format(i))
            viewer_generated = last_image

        frame = cv2.resize(frame, (1024, 1024))
        viewer_generated = np.concatenate((viewer_generated, frame), axis=0)
        cv2.imwrite("images/output_{}.png".format(i), viewer_generated)
        last_image = viewer_generated
        #cv2.imshow("Preview", viewer_generated)
        if cv2.waitKey(10) == 27:
            break
        i += 1

    # Clean up the multiprocessing process.
    video_capture.release()
    frames_to_video.create_video_from_frames(dir_path='images', ext='png', output=save_file)



if __name__ == '__main__':
    for inv_fn, inv in {'neural_1_3': ['models/inverter/inverter_neural_3_8000_good.pkl', 'models/inverter/inverter_neural_1_8000_good.pkl']}.items():
        #video_capture = cv2.VideoCapture('cam_video2.mp4')
        #process_video_capture(video_capture, save_file='videos/video_cam_video2_{}.mp4'.format(inv_fn), inverter_filenames=inv)
        video_capture = cv2.VideoCapture('cam_video.mp4')
        process_video_capture(video_capture, save_file='videos/video_cam_video_{}.mp4'.format(inv_fn), inverter_filenames=inv)
