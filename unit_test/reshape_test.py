
import numpy as np
from PIL import Image
import tensorflow as tf

img_name = "../data/coco/train2014/COCO_train2014_000000465294.jpg"

img = np.array(Image.open(img_name))

print(img.shape)

im_shape = tf.shape(img)

print(im_shape)

print(im_shape[0])

image = tf.reshape(img, (im_shape[0], im_shape[1], im_shape[2], 3))

print(image)