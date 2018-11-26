import tensorflow as tf
import cv2
import random

def union(a, b, area_intersect):
  area_a = (a[2] - a[0]) * (a[3] - a[1])
  area_b = (b[2] - b[0]) * (b[3] - b[1])
  area_union = area_a + area_b - area_intersect
  return area_union

def intersetion(a, b):
  x1 = tf.maximum(a[0], b[0])
  y1 = tf.maximum(a[1], b[1])
  x2 = tf.maximum(a[2], b[2])
  y2 = tf.maximum(a[3], b[3])
  w = x2 - x1
  h = y2 - y1
  if w <= 0 or h < 0:
    return 0
  return w * h

def iou(a, b):
  # (x1, y1, x2, y2)
  area_intersect = intersetion(a, b)
  area_union = union(a, b, area_intersect)
  return area_intersect / area_union

def get_new_img_size(width, height, img_min_side=600):
  if width <= height:
    f = img_min_side / width
    resized_height = (f * height).as_type(tf.int32)
    resized_width = img_min_side
  else:
    f = img_min_side / height
    resized_width = (f * width).as_type(tf.int32)
    resized_height = img_min_side
  return resized_width, resized_height