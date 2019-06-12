import serial
import struct
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
import time
import cv2

def gstreamer_pipeline (capture_width=500, capture_height=500, display_width=500,
     display_height=500, framerate=60, flip_method=0) :
     return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def XZ_angle_posX(XZ):
        if XZ[1] == 0 and XZ[0] == 0:
            return 2.0 * np.pi
        if XZ[1] == 0 and XZ[0] > 0:
            return 0.0
        elif XZ[1] == 0 and XZ[0] < 0:
            return np.pi
        elif XZ[0] == 0 and XZ[1] > 0:
            return np.pi / 2.0
        elif XZ[0] == 0 and XZ[1] < 0:
            return (np.pi / 2.0) * 3.0
        elif XZ[0] > 0 and XZ[1] > 0:
            return np.arctan(XZ[1] / XZ[0])
        elif XZ[0] < 0 and XZ[1] > 0:
            return np.pi + np.arctan(XZ[1] / XZ[0])
        elif XZ[0] < 0 and XZ[1] < 0:
            return np.pi + np.arctan(XZ[1] / XZ[0])
        elif XZ[0] > 0 and XZ[1] < 0:
            return 2.0 * np.pi + np.arctan(XZ[1] / XZ[0])

def acquire_cam_obs(img):
    img = cv2.flip(img, 0)
    img = cv2.flip(img, 1)
    img = cv2.blur(img, (3,3))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0, 70, 50), (10, 255, 255))
    mask2 = cv2.inRange(img_hsv, (170, 70, 50), (180, 255, 255))
    mask = mask1 | mask2
    #mask = cv2.bitwise_not(mask)
    cv2.imshow("Image1", mask)
    keycode = cv2.waitKey(30) & 0xff
    M = cv2.moments(mask)
    if M["m00"] == 0 or M["m00"] == 0:
        return [0, 0]
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cX = cX - (500/2.0)
    cY = (500/2.0) - cY
    norm_pt = [cX / (500/2), cY / (500/2)]
    XZ_angle = XZ_angle_posX(norm_pt)
    dist = np.linalg.norm(norm_pt)
    return norm_pt, XZ_angle, dist

ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=None)
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], '/home/aslan/Downloads/tf_ppo2_prune1_3')
        obs = graph.get_tensor_by_name("input/Ob:0")
        action = graph.get_tensor_by_name("output/add:0")
        yaw_in = 0.0
        pitch_in = 0.0
        TRANSMIT_OK = False
        RECEIVE_OK = True
        while (1):
            if RECEIVE_OK and ser.in_waiting == 4:
                arr = [0, 0, 0, 0]
                for num in range(4):
                    arr[num] = struct.unpack('b', ser.read(1))[0]
                    yaw_in = float((arr[0]) << 8 | (arr[1] & 0xFF)) / 1000.0
                    pitch_in = float((arr[2] & 0xFF) << 8 | (arr[3] & 0xFF)) / 1000.0
                TRANSMIT_OK = True
                RECEIVE_OK = False
            if TRANSMIT_OK:
                ret_val, img = cap.read()
                XZ, XZ_angle, dist = acquire_cam_obs(img)
                obser = np.array([yaw_in, pitch_in, XZ[0], XZ[1], XZ_angle, dist])
                obser = obser.reshape((-1,) + (6,))
                target_angular_vel = sess.run(action, feed_dict={obs:obser})
                target_angular_vel[0][0] = min(10, abs(target_angular_vel[0][0]))*(abs(target_angular_vel[0][0])/target_angular_vel[0][0])
                target_angular_vel[0][1] = min(5, abs(target_angular_vel[0][1]))*(abs(target_angular_vel[0][1])/target_angular_vel[0][1])
                target_bytes = bytearray([np.uint8(int(target_angular_vel[0][0]*1000) >> 8), np.uint8(int(target_angular_vel[0][0]*1000) & 0xFF), np.uint8(int(target_angular_vel[0][1]*1000) >> 8), np.uint8(int(target_angular_vel[0][1]*1000) & 0xFF)])
                print(target_angular_vel)
                if ser.write(target_bytes) == 4:
                    TRANSMIT_OK = False
                    RECEIVE_OK = True


