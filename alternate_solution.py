import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

class Model:
    def __init__(self, include_top, weights, input_shape):
        assert input_shape == (299,299,3)
        self.model = InceptionV3(
            include_top=include_top,
            weights=weights, 
            input_shape=input_shape
        )
        self.height = input_shape[0]
        self.width = input_shape[1]
        print("Model is initialized successfully!")

    def preprocess_frame(self, img):
        img = cv2.resize(img, (self.height, self.width))
        img = img.astype(np.float32)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img
    
    def __call__(self, x):
        y = self.model(x)
        return y
    
    def predict(self, x):
        y = self.model(x)
        return y
    
    def textify(self, y, labels):
        y = y.numpy().flatten()
        prob = y[np.argmax(y)]
        percent = "{:.2f}%".format(prob*100)
        text_color = (0,0,255) if prob < 0.8 else (0,255,0)
        text_label = str(percent) + 2*" " + labels[np.argmax(y)]
        return text_label, text_color

def read_label(labelpath):
    with open(labelpath) as f:
        lines = f.readlines()
    labels = [l.strip() for l in lines]
    labels.remove("dummy")
    return labels

def draw_frame(frame, text_label, text_color):
    # put text
    x, y = 5, 20
    cv2.putText(frame, text_label , (x, y), 2, 0.6, text_color, 1)
    
    # add black border
    pad = 50
    pad_color = (0,0,0)
    top = bottom = left = right = 50
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, pad_color)
    
    # draw filled rectange
    x1, y1, x2, y2 = 50, H+60, W+50, H+90
    fill_color = (103, 103, 28)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color , -1)
    
    # put text on filled rectangle
    x, y = 55, H+80
    cv2.putText(frame, text_label , (x, y), 2, 0.6, text_color, 1)
    
    return frame

if __name__ == '__main__':

    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="Path to the input video")
    parser.add_argument("-l", "--labelpath", help="Path to the class labels")
    parser.add_argument("-w", "--weights", help="Path to the model weights .h5")
    args = parser.parse_args()
    
    source = args.source if args.source else os.path.join(os.getcwd(), 'data', 'sample.mp4')
    labelpath = args.labelpath if args.labelpath else os.path.join(os.getcwd(), 'data', 'imagenet_slim_labels.txt')
    weights = args.weights if args.weights else 'imagenet'

    # Read labels
    labels = read_label(labelpath)

    # Initilaize Model object
    include_top = True
    input_shape=(299,299,3)
    model = Model(include_top, weights, input_shape)

    cap = cv2.VideoCapture(source)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        x = model.preprocess_frame(frame)
        y = model(x)
        text_label, text_color = model.textify(y, labels)
        frame = draw_frame(frame, text_label, text_color)
    
        cv2.imshow("Window", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()