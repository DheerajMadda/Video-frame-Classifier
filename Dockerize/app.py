import os
import cv2
import tempfile
import streamlit as st
from utils import *


# Read labels
labelpath = os.path.join(os.getcwd(), 'data', 'imagenet_slim_labels.txt')
labels = read_label(labelpath)

# Initilaize Model object
include_top = True
input_shape=(299,299,3)
weights = os.path.join(os.getcwd(), 'data', 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
model = Model(include_top, weights, input_shape)

if __name__ == "__main__":

    st.title("Video Frame Classifier")
    st.subheader("Home")
    VIDEO_EXTENSIONS = ["mp4", "ogv", "m4v", "webm"]
    video_file = st.file_uploader("Upload the video", type=VIDEO_EXTENSIONS)

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            x = model.preprocess_frame(frame)
            y = model(x)
            text_label, text_color = model.textify(y, labels)
            frame = draw_frame(frame, text_label, text_color, (H, W))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame)