
# Video frame Classifier
![video](https://user-images.githubusercontent.com/50489165/148655338-95f55a13-f42c-4d3a-b7c4-6c5b7d52ea19.gif)

### docker pull dheerajmadda/videoframeclassifier
### docker run -p 8080:8080 dheerajmadda/videoframeclassifier



###  Description

The aim of this project is to take a video as an input and to display the classified label with confidence (in percentage).
Make sure you have necessary libraries installed :- opencv-python, numpy, tensorflow

There are 2 solutions that are implemented using different approaches.


### To run
A) Clone the repo 

B) Follow the below steps:

### Solution 1) Using tensorflow graph api (solution.py) [This is based on the repo https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image]

Download https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz and extract it into the data folder

python3 solution.py -s data/sample.mp4 -l data/imagenet_slim_labels.txt -m data/inception_v3_2016_08_28_frozen.pb

#### for custom video:
python3 solution.py -s <video_path> -l <label_path> -m <model_path>

### Solution 2) Using tensorflow keras api (alternate_solution.py)

Download: https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5 and move it into the data folder

python3 alternate_solution.py -s data/sample.mp4 -l data/imagenet_slim_labels.txt -w data/inception_v3_weights_tf_dim_ordering_tf_kernels.h5

#### for custom video:
python3 alternate_solution.py -s  <video_path> -l <label_path> -w <weights_path>


## Want to build an image? 

1) Clone the repo -> cd Dockerize
2) Download: https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5 and move it into the data folder (i.e. Dockerize/data/)
3) docker build -t <image_name> .

#### Run as a container
😊 docker run -p 8080:8080 <image_name>


