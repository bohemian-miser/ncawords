import tensorflow as tf
from run_single import CAModel, make_seed, create_target_image, to_rgba
import numpy as np

target_img = create_target_image()
pad_target = tf.pad(target_img, [(8, 8), (8, 8), (0, 0)])
h, w = pad_target.shape[:2]
seed = make_seed(h, w, 1)

ca = CAModel()

def loss_f(x):
    return tf.reduce_mean(tf.square(to_rgba(x) - pad_target))

with tf.GradientTape() as g:
    x = tf.constant(seed)
    for i in range(4):
        x = ca(x)
    loss = tf.reduce_mean(loss_f(x))

grads = g.gradient(loss, ca.weights)
for i, gr in enumerate(grads):
    print(f"ca.weights[{i}] (shape {ca.weights[i].shape}) grad norm:", np.linalg.norm(gr) if gr is not None else "None")

