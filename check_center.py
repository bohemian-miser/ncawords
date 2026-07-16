from run_single import create_target_image
import tensorflow as tf
import numpy as np

target_img = create_target_image("C", size=(40, 40))
pad_target = tf.pad(target_img, [(8, 8), (8, 8), (0, 0)])
h, w = pad_target.shape[:2]
center_pixel = pad_target[h//2, w//2, 3].numpy()
print("Center pixel alpha for 'C':", center_pixel)

target_img_word = create_target_image("COMP6441", size=(128, 40))
pad_target_word = tf.pad(target_img_word, [(8, 8), (8, 8), (0, 0)])
h_w, w_w = pad_target_word.shape[:2]
center_pixel_word = pad_target_word[h_w//2, w_w//2, 3].numpy()
print("Center pixel alpha for 'COMP6441':", center_pixel_word)
