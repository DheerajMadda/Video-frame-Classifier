import os
import cv2
import argparse
import numpy as np
import tensorflow as tf

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef() if is_tfv2 else tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def load_labels(label_file):
    if is_tfv2:
        proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    else:
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    return [l.rstrip() for l in proto_as_ascii_lines]

def read_tensor_from_image_file(frame,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):

    float_caster = tf.cast(frame, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    if is_tfv2:
        resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    else:
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    return sess.run(normalized)

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

def textify(y, labels):
    prob = y[np.argmax(y)]
    percent = "{:.2f}%".format(prob*100)
    text_color = (0,0,255) if prob < 0.8 else (0,255,0)
    text_label = str(percent) + 2*" " + labels[np.argmax(y)]
    return text_label, text_color


if __name__ == '__main__':

    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="Path to the input video")
    parser.add_argument("-l", "--labelpath", help="Path to the class labels")
    parser.add_argument("-m", "--model", help="Path to the model")
    args = parser.parse_args()
    
    source = args.source if args.source else os.path.join(os.getcwd(), 'data', '2.mp4')
    label_file = args.labelpath if args.labelpath else os.path.join(os.getcwd(), 'data', 'imagenet_slim_labels.txt')
    model_file = args.model if args.model else os.path.join(os.getcwd(), 'data', 'inception_v3_2016_08_28_frozen.pb')

    # check tensorflow version
    is_tfv2 = True if int(tf.__version__[0]) == 2 else False
    if is_tfv2: tf.compat.v1.disable_eager_execution()

    # define required parameters
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer

    # Load graph
    graph = load_graph(model_file)
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    # Read labels
    labels = load_labels(label_file)

    cap = cv2.VideoCapture(source)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        x = read_tensor_from_image_file(
            frame,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)
    
        with tf.compat.v1.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: x
            })

        y = np.squeeze(results)
        text_label, text_color = textify(y, labels)
        frame = draw_frame(frame, text_label, text_color)
    
        cv2.imshow("Window", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()