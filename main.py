#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time

from skimage import transform as sktf
from skimage import exposure
import skimage
import random
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
import scipy.misc

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path) # load a graph from the file
    graph = tf.get_default_graph() # get the graph

    # get tensors from the graph
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    ker_reg = tf.contrib.layers.l2_regularizer(1e-3) # kernel regularizer
    ker_init = tf.initializers.truncated_normal(mean = 0.0, stddev = 1e-3, seed = 1) # kernel initializer

    # scaling
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001)
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01)

    # 1x1 convolutions
    conv_1x1_layer7 = tf.layers.conv2d(inputs = vgg_layer7_out, filters = num_classes,
                                    kernel_size = 1, strides = (1, 1), padding='same',
                                        kernel_regularizer = ker_reg, kernel_initializer = ker_init)

    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out_scaled, num_classes, 1, (1, 1), 'same',
                                        kernel_regularizer = ker_reg, kernel_initializer = ker_init)

    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out_scaled, num_classes, 1, (1, 1), 'same',
                                        kernel_regularizer = ker_reg, kernel_initializer = ker_init)

    # deconvolutions (upsampling) with skip connections
    deconv_layer7 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, 4, (2, 2), padding='same',
                                        kernel_regularizer = ker_reg, kernel_initializer = ker_init)

    skip_conv_1x1_layer4 = tf.add(deconv_layer7, conv_1x1_layer4)
    deconv_layer4 = tf.layers.conv2d_transpose(skip_conv_1x1_layer4, num_classes, 4, (2, 2), padding='same',
                                        kernel_regularizer = ker_reg, kernel_initializer = ker_init)

    skip_conv_1x1_layer3 = tf.add(deconv_layer4, conv_1x1_layer3)
    nn_last_layer = tf.layers.conv2d_transpose(skip_conv_1x1_layer3, num_classes, 16, (8, 8), padding='same',
                                        kernel_regularizer = ker_reg, kernel_initializer = ker_init)

    return nn_last_layer

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    normal_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = correct_label, logits = logits))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy_loss_reg = normal_loss + sum(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss_reg)

    return logits, train_op, cross_entropy_loss_reg

tests.test_optimize(optimize)


def data_aug(images):
    """
    This function is for data augumentation with rescale intensity and gamma correction.
    :param images: batch of input images
    :param labels: batch of labels (note that a label is a segmentation image)
    :return: tuple of augmented images and corresponding labels

    """
    images_aug = []

    for i in range(len(images)):
        image = images[i]

        # apply rescale intensity and gamma correction
        rand_gamma = random.uniform(0.5, 1.0)
        rand_gain = random.uniform(0.5, 1.0)
        lower_p = random.uniform(0, 20)
        upper_p = random.uniform(80, 100)
        v_min, v_max = np.percentile(image, (lower_p, upper_p))
        image_aug = exposure.rescale_intensity(image, in_range=(v_min, v_max))
        image_aug = exposure.adjust_gamma(image_aug, gamma=rand_gamma, gain=rand_gain)

        images_aug.append(image_aug)

    return np.array(images_aug)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # initialization
    sess.run(tf.global_variables_initializer())

    prob = 0.5 # keep probability for the dropout layer
    lr = 0.00007 # learning rate
    file_name = 'loss.txt' # file to save values of losses
    f = open(file_name, 'w')

    epoch = 0
    print("----- training started! -----")
    for i in range(epochs):
        start_time = time.time()
        epoch += 1
        for image, label in get_batches_fn(batch_size):

            # # in case of doing data augmentation, use below
            # # (note: if using this, the code for the test in "project_tests.py"
            # #  gives an error. Please comment out `tests.test_train_nn(train_nn)` below.
            # image = data_aug(image)

            _, ce_loss = sess.run([train_op,cross_entropy_loss],
                                    feed_dict = {input_image: image, correct_label: label, keep_prob: prob, learning_rate: lr})

        time_elapsed = time.time() - start_time
        # print the loss and elasped time, write the loss to the text file
        print("epoch {}: loss {}, elapsed time {}".format(epoch, ce_loss, time_elapsed))
        f.write("{} {}\n".format(epoch, ce_loss))

    f.close()
    print("----- training done! -----")

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    batch_size = 8
    epochs = 40 

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

        # train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                                                input_image, correct_label, keep_prob, learning_rate)

        # save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # for saving the video with the model applied for the segmentation
        save_inference_video(sess, image_shape, logits, keep_prob, input_image, video_file="videos/test.mp4")


def save_inference_video(sess, image_shape, logits, keep_prob, input_image, video_file):
    """
    This function applies the trained model to a mp4 video
    :param video_file: path to the input mp4 video file
    :param sess: TF session
    :param image_shape: Tuple - Shape of image
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param input_image: TF Placeholder for the image placeholder
    """
    # create list of images from mp4
    def gen_test_output_image(image):
        """
        Generate test output using the test images
        :param image: image in numpy array
        :return: Output for test image
        """
        image_resize = scipy.misc.imresize(image, image_shape)
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image_resize]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image_resize)
        street_im.paste(mask, box=None, mask=mask)

        return np.array(street_im)

    # apply gen_test_output_image to each frame of the mp4 video
    clip_test = VideoFileClip(video_file)
    clip_test_segmented = clip_test.fl_image(gen_test_output_image)
    output_file = video_file[:-4]+"_segmented.mp4"
    clip_test_segmented.write_videofile(output_file, audio=False)


def plot_loss(file_name = "loss.txt"):
    """
    This function is used to create a png file plotting losses.
    : param file_name: file name of the text file storing losses
    """

    # read a data from the text file
    with open(file_name) as f:
        data = f.read()

    # formatting the data
    data = data.split("\n")[:-1]
    epochs = [int(row.split(" ")[0]) for row in data]
    losses = [float(row.split(" ")[1]) for row in data]

    # plot and save as a png file
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("# of epochs vs loss")
    ax1.set_xlabel('# of epochs')
    ax1.set_ylabel('loss')
    ax1.plot(epochs, losses)
    plt.savefig("loss.png")

if __name__ == '__main__':
    run()
    plot_loss()
