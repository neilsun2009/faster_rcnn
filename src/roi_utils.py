# roi utilities

import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import data_generator

def delete_np(tensor, idx, axis=0):
  return np.delete(tensor, idx, axis=axis)

def cal_iou(rois, img_data, config, class_mapping):
  bboxes = img_data['bboxes']
  width, height = img_data['width'], img_data['height']
  resized_width, resized_height = data_generator.get_new_img_size(width, height, config.img_size)
  
  # ground truth
  gt = tf.zeros((len(bboxes), 4))
  for idx, bbox in enumerate(bboxes):
    gt[idx, 0] = (bbox['x1'] * (resized_width/width) / config.rpn_stride).as_type(tf.int32)
    gt[idx, 2] = (bbox['x2'] * (resized_width/width) / config.rpn_stride).as_type(tf.int32)
    gt[idx, 1] = (bbox['y1'] * (resized_height/height) / config.rpn_stride).as_type(tf.int32)
    gt[idx, 3] = (bbox['y2'] * (resized_height/height) / config.rpn_stride).as_type(tf.int32)

  x_roi = []
  y_class_num = []
  y_class_regr_coords = []
  y_class_regr_label = []
  ious = [] # for debug

  for i in range(rois.shape[0]):
    x1, y1, x2, y2 = rois[i, :]
    best_iou = 0
    best_bbox = -1
    for idx in range(len(bboxes)):
      curr_iou = data_generator.iou((gt[idx, 0], gt[idx, 1], gt[idx, 2], gt[idx, 3]),
        (x1, y1, x2, y2))
      if curr_iou > best_iou:
        best_iou = curr_iou
        best_bbox = idx
    if best_iou < config.classifier_min_overlap:
      continue
    w = x2 - x1
    h = y2 - y1
    x_roi.append([x1, y1, w, h])
    ious.append(best_iou)
    # check for ground truth class and regression delta value 
    if best_iou < config.classifier_max_overlap:
      cls_name = 'bg'
    else:
      cls_name = bbox[best_bbox]['class']
      cx_gt = (gt[best_bbox, 0] + gt[best_bbox, 2]) / 2
      cy_gt = (gt[best_bbox, 1] + gt[best_bbox, 3]) / 2
      cx = (x1 + x2) / 2
      cy = (y1 + y2) / 2
      tx = (cx_gt - cx) / w
      ty = (cy_gt - cy) / h
      tw = tf.math.log((gt[best_bbox, 2] - gt[best_bbox, 0]) / w)
      th = tf.math.log((gt[best_bbox, 3] - gt[best_bbox, 1]) / h)
    # order
    class_num = class_mapping[cls_name]
    class_label = len(class_mapping) * [0]
    class_label[class_num] = 1
    y_class_num.append(class_label.copy())
    coords = [0] * 4 * (len(class_mapping) - 1) # remove bg class
    labels = [0] * 4 * (len(class_mapping) - 1)
    if cls_name != 'bg':
      label_pos = 4 * class_num
      sx, sy, sw, sh = config.classifier_regr_std
      coords[label_pos:label_pos+4] = [sx*tx, sy*ty, sw*tw, sh*th]
      coords[label_pos:label_pos+4] = [1, 1, 1, 1]
      y_class_regr_coords.append(coords.copy())
      y_class_regr_label.append(labels.copy())
    else:
      y_class_regr_coords.append(coords.copy())
      y_class_regr_label.append(labels.copy())
  
  # return
  if len(x_roi) == 0:
    return None, None, None, None
  X = tf.expand_dims(x_roi, axis=0)
  Y1 = tf.expand_dims(y_class_num, 0)
  Y2 = tf.expand_dims(tf.concat((y_class_regr_label, y_class_regr_coords), axis=1), 0)
  return X, Y1, Y2, ious
    
    
def non_max_suppresion_fast(boxes, probs, overlap_threshold=0.9, max_boxes=300):
  # code used refers to here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
  if len(boxes) == 0:
    return []
  
  boxes = tf.cast(boxes, tf.float32)
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]

  pick = []
  area = (x2 - x1) * (y2 - y1)
  idxs = tf.contrib.framework.argsort(boxes)

  while len(idxs) > 0:
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    xx1 = tf.maximum(x1[i], x1[idxs[:last]])
    yy1 = tf.maximum(y1[i], y1[idxs[:last]])
    xx2 = tf.minimum(x2[i], x2[idxs[:last]])
    yy2 = tf.minimum(y2[i], y2[idxs[:last]])

    ww = tf.maximum(0, xx2-xx1)
    hh = tf.maximum(0, yy2-yy1)
    area_intersect = ww * hh

    area_union = area[i] + area[idxs[:last]] - area_intersect
    overlap = area_intersect / (area_union + 1e-6)

    idxs = tf.py_func(delete_np, [idxs, tf.concat(([last], tf.where(overlap > overlap_threshold)))], [idxs.dtype])

    if len(pick) >= max_boxes:
      break
  
  boxes = boxes[pick].as_type(tf.int32)
  probs = probs[pick]
  return boxes, probs

def apply_regress(X, T):
  try: 
    x = X[0, :, :]
    y = X[1, :, :]
    w = X[2, :, :]
    h = X[3, :, :]

    tx = T[0, :, :]
    ty = T[1, :, :]
    tw = T[2, :, :]
    th = T[3, :, :]

    new_w =tf.exp(tw) * w
    new_h =tf.exp(th) * h
    new_x = tx*w + x + w/2 - new_w/2
    new_y = ty*h + y + h/2 - new_h/2

    new_x = tf.round(new_x)
    new_y = tf.round(new_y)
    new_w = tf.round(new_w)
    new_h = tf.round(new_h)
    return tf.stack([new_x, new_y, new_w, new_h])
  except Exception as e:
    print(e)
    return X


def rpn_to_roi(rpn_layer, regress_layer, config, max_boxes=300,
  overlap_threshold=0.9):
  regress_layer = regress_layer / config.std_scaling

  anchor_sizes = config.anchor_box_scales
  anchor_ratios = config.anchor_box_ratios

  assert rpn_layer.shape[0] == 1
  rows, cols = rpn_layer.shape[1:3]

  curr_layer = 0
  A = tf.zeros((4, rows, cols, rpn_layer.shape[3])) # xywh (anchor feature) for each pixel for each anchor

  for anchor_size in anchor_sizes:
    for anchor_ratio in anchor_ratios:
      # rpn stride, the downsample rate frm original image to feature
      anchor_x = (anchor_size * anchor_ratio[0]) / config.rpn_stride
      anchor_y = (anchor_size * anchor_ratio[1]) / config.rpn_stride
      regress = regress_layer[0, :, :, 4*curr_layer:4*(curr_layer+1)]
      regress = tf.transpose(regress, (2, 0, 1)) # xywh, x, y
      
      X, Y = tf.meshgrid(tf.range(cols), tf.range(rows))

      # original xyhw
      A[0, :, :, curr_layer] = X - anchor_x / 2
      A[1, :, :, curr_layer] = Y - anchor_y / 2
      A[2, :, :, curr_layer] = anchor_x
      A[3, :, :, curr_layer] = anchor_y

      # adjust according to rpn regression result
      A[:, :, :, curr_layer] = apply_regress(A[:, :, :, curr_layer], regress)

      # h/w restriction and convert h/w to y2/x2
      A[2, :, :, curr_layer] = tf.math.maximum(1, A[2, :, :, curr_layer])
      A[3, :, :, curr_layer] = tf.math.maximum(1, A[3, :, :, curr_layer])
      A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
      A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

      # border restriction
      A[0, :, :, curr_layer] = tf.math.maximum(0, A[0, :, :, curr_layer])
      A[1, :, :, curr_layer] = tf.math.maximum(0, A[1, :, :, curr_layer])
      A[2, :, :, curr_layer] = tf.math.minimum(cols-1, A[2, :, :, curr_layer])
      A[3, :, :, curr_layer] = tf.math.minimum(rows-1, A[3, :, :, curr_layer])

      curr_layer += 1
  
  all_boxes = tf.reshape(tf.transpose(A, (0, 3, 1, 2)), (4, -1))
  all_boxes = tf.transpose(all_boxes, (1, 0))
  all_probs = tf.transpose(rpn_layer, (0, 3, 1, 2))
  all_probs = tf.reshape(all_probs, (-1))

  x1 = all_boxes[:, 0]
  y1 = all_boxes[:, 1]
  x2 = all_boxes[:, 2]
  y2 = all_boxes[:, 3]

  del_idx = tf.where((x1-x2>=0 | (y1-y2>=0)))

  all_boxes = tf.py_func(delete_np, [all_boxes, del_idx, 0], [all_boxes.dtype])
  all_probs = tf.py_func(delete_np, [all_probs, del_idx, 0], [all_probs.dtype])

  result = non_max_suppresion_fast(all_boxes, all_probs, overlap_threshold=overlap_threshold,
    max_boxes=max_boxes)[0]
  return result