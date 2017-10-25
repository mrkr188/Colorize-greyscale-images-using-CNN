import sys
import tensorflow as tf
import numpy as np
import scipy.misc
from glob import glob
from utils import read_my_file_format
from utils import input_pipeline
from utils import concat_images
from utils import rgb2yuv
from utils import yuv2rgb
from batchnorm import batch_norm

# store all test images filenames
filenames = sorted(glob("*.jpg"))

# batch_size and number of epochs
batch_size = 1
num_epochs = 1000000

# variable creation
global_step = tf.Variable(0, name='global_step', trainable=False)
phase_train = tf.placeholder(tf.bool, name='phase_train')
uv = tf.placeholder(tf.uint8, name='uv')


# a helper conv2d function
def conv2d(X, w, sigmoid=False, bn=False):
    with tf.variable_scope('conv2d'):
        X = tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME')
        if bn:
            X = batch_norm(X, w.get_shape()[3])
        if sigmoid:
            return tf.sigmoid(X)
        else:
            X = tf.nn.relu(X)
            return tf.maximum(0.01 * X, X)


# Resnet inspired network to predict U and V channels
def colorize_net(_tensors):
    with tf.variable_scope('colorize_net'):

        # the conv5_3 layer of vgg is batch normalized, and 1x1 conv is applied on it
        # Bx14x14x512 -> batch norm -> 1x1 conv = Bx14x14x512
        conv0 = conv2d(batch_norm(_tensors["conv5_3"], 512), _tensors["weights"]['wc0'], sigmoid=False, bn=True)
        # upscale the resulting layer to 28x28x512
        conv0 = tf.image.resize_bilinear(conv0, (28, 28))
        # add the layer to conv4_3 vgg layer
        conv0 = tf.add(conv0, batch_norm(_tensors["conv4_3"], 512))

        # the conv4_3 layer of vgg is batch normalized, and 1x1 conv is applied on it
        # Bx28x28x512 -> 1x1 conv = Bx28x28x256
        conv1 = conv2d(conv0, _tensors["weights"]['wc1'], sigmoid=False, bn=True)
        # upscale the resulting layer to 56x56x256
        conv1 = tf.image.resize_bilinear(conv1, (56, 56))
        # add the upscaled layer to conv3_3 vgg layer
        conv1 = tf.add(conv1, batch_norm(_tensors["conv3_3"], 256))

        # Bx56x56x256 -> 3x3 conv = Bx56x56x128
        conv2 = conv2d(conv1, _tensors["weights"]['wc2'], sigmoid=False, bn=True)
        # upscale the resulting layer to 112x112x128
        conv2 = tf.image.resize_bilinear(conv2, (112, 112))
        # add the upscaled layer to conv3_3 vgg layer
        conv2 = tf.add(conv2, batch_norm(_tensors["conv2_2"], 128))
        
        # Bx112x112x128 -> 3x3 conv = Bx112x112x64
        conv3 = conv2d(conv2, _tensors["weights"]['wc3'], sigmoid=False, bn=True)
        # upscale the resulting layer to Bx224x224x64
        conv3 = tf.image.resize_bilinear(conv3, (224, 224))
        # add the upscaled layer to conv1_2 vgg layer
        conv3 = tf.add(conv3, batch_norm(_tensors["conv1_2"], 64))

        # Bx224x224x64 -> batch norm -> 3x3 conv = Bx224x224x3
        conv4 = conv2d(conv3, _tensors["weights"]['wc4'], sigmoid=False, bn=True)
        # add the layer to greyscale image layer
        conv4 = tf.add(conv4, batch_norm(_tensors["grayscale"], 3))

        # Bx224x224x3 -> 3x3 conv = Bx224x224x3
        conv5 = conv2d(conv4, _tensors["weights"]['wc5'], sigmoid=False, bn=True)

        # Bx224x224x3 -> 3x3 conv = Bx224x224x2
        conv6 = conv2d(conv5, _tensors["weights"]['wc6'], sigmoid=True, bn=True)

    return conv6


# import vgg model
with open("vgg16-20160129.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

# Initialize and store weights. normal distribution is recommended for initializing weights.
with tf.variable_scope('colorize_net'):
    weights = {
        # 1x1 conv, 512 inputs, 512 outputs
        'wc0': tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev=0.01)),
        # 1x1 conv, 512 inputs, 256 outputs
        'wc1': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01)),
        # 3x3 conv, 512 inputs, 128 outputs
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01)),
        # 3x3 conv, 256 inputs, 64 outputs
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01)),
        # 3x3 conv, 128 inputs, 3 outputs
        'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01)),
        # 3x3 conv, 6 inputs, 3 outputs
        'wc5': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01)),
        # 3x3 conv, 3 inputs, 2 outputs
        'wc6': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01)),
    }

# pre process input image
reader = tf.WholeFileReader()

colorimage = input_pipeline(filenames, batch_size, num_epochs=num_epochs)
colorimage_yuv = rgb2yuv(colorimage)

grayscale = tf.image.rgb_to_grayscale(colorimage)
grayscale_rgb = tf.image.grayscale_to_rgb(grayscale)
grayscale_yuv = rgb2yuv(grayscale_rgb)
grayscale = tf.concat(3, [grayscale, grayscale, grayscale])


# build tensorflow graph
tf.import_graph_def(graph_def, input_map={"images": grayscale})

graph = tf.get_default_graph()

# get weights from vgg pre trained model
with tf.variable_scope('vgg'):
    conv1_2 = graph.get_tensor_by_name("import/conv1_2/Relu:0")
    conv2_2 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
    conv3_3 = graph.get_tensor_by_name("import/conv3_3/Relu:0")
    conv4_3 = graph.get_tensor_by_name("import/conv4_3/Relu:0")
    conv5_3 = graph.get_tensor_by_name("import/conv5_3/Relu:0")


# create dictionary of input tensors
tensors = {
    # the vgg layers
    "conv1_2": conv1_2,
    "conv2_2": conv2_2,
    "conv3_3": conv3_3,
    "conv4_3": conv4_3,
    "conv5_3": conv5_3,
    # input greyscale image
    "grayscale": grayscale,
    # the colorize_net weights to predict UV channels
    "weights": weights
}

# Construct model
pred = colorize_net(tensors)
pred_yuv = tf.concat(3, [tf.split(3, 3, grayscale_yuv)[0], pred])
pred_rgb = yuv2rgb(pred_yuv)

init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

# start session
sess = tf.Session()

sess.run(init_op)

# import model
saver = tf.train.import_meta_graph('full_model/colorize_model_2.ckpt.meta')
saver.restore(sess, 'full_model/colorize_model_2.ckpt.data-00000-of-00001')

pred_, pred_rgb_, colorimage_, grayscale_rgb_ = sess.run(
       [pred, pred_rgb, colorimage, grayscale_rgb], feed_dict={phase_train: False})

summary_image = concat_images(grayscale_rgb_[0], pred_rgb_[0])
summary_image = concat_images(summary_image, colorimage_[0])
# save color output
scipy.misc.imsave("summary_1.jpg", colorimage_[0])
# save summary image
scipy.misc.imsave("summary_1.jpg", summary_image)

sess.close()
