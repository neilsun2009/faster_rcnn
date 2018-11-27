import tensorflow as tf
import numpy as np
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

def data_augment(img_data, config):
  assert 'filepath' in img_data
  assert 'bboxes' in img_data
  assert 'width' in img_data
  assert 'height' in img_data

  img_data_aug = img_data.copy()
  img = tf.read_file(img_data_aug['filepath'])
  img = tf.image.decode_jpeg(img, channels=3)

  height, width = tf.shape(img)[:2]
  if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
    img = tf.image.flip_left_right(img)
    for bbox in img_data_aug['bboxes']:
      x1 = bbox['x1']
      x2 = bbox['x2']
      bbox['x2'] = width - x1
      bbox['x1'] = width - x2

  if config.use_vertical_flips and np.random.randint(0, 2) == 0:
    img = tf.image.flip_up_down(img)
    for bbox in img_data_aug['bboxes']:
      y1 = bbox['y1']
      y2 = bbox['y2']
      bbox['y2'] = height - y1
      bbox['y1'] = height - y2

  if config.rot_90:
    angle = np.random.choice([0, 90, 180, 270], 1)[0]
    if angle == 270:
      img = tf.image.transpose_image(img)
      img = tf.image.flip_up_down(img)
    elif angle == 180:
      img = tf.image.flip_up_down(img)
      img = tf.image.flip_left_right(img)
    elif angle == 90:
      img = tf.image.transpose_image(img)
      img = tf.image.flip_left_right(img)
    elif angle == 0:
      pass
    
    for bbox in img_data_aug['bboxes']:
      x1 = bbox['x1']
      x2 = bbox['x2']
      y2 = bbox['y2']
      y1 = bbox['y1']
      if angle == 270: 
        bbox['x1'] = y1
        bbox['x2'] = y2
        bbox['y1'] = width - x2
        bbox['y2'] = width - x1
      elif angle == 180:
        bbox['x2'] = width - x1
        bbox['x1'] = width - x2
        bbox['y2'] = height - y1
        bbox['y1'] = height - y2
      elif angle == 90:
        bbox['x1'] = height - y2
        bbox['x2'] = height - y1
        bbox['y1'] = x1
        bbox['y2'] = x2
      elif angle == 0:
        pass
  img_data_aug['width'] = tf.shape(img)[1]
  img_data_aug['height'] = tf.shape(img)[0]
  return img_data_aug, img
  
def cal_rpn(config, img_data, width, height, resized_width, resized_height, img_length_calc_func):
  downscale = config.rpn_stride
  anchor_sizes = config.anchor_box_scales
  anchor_ratios = config.anchor_box_ratios
  n_anchratios = len(anchor_ratios)
  num_anchors = n_anchratios  * len(anchor_sizes)

  # feature w/h
  (output_width, output_height) = img_length_calc_func(resized_width, resized_height)
  
  # init output
  y_rpn_overlap = tf.zeros((output_height, output_width, num_anchors))
  y_is_box_valid = tf.zeros((output_height, output_width, num_anchors))
  y_rpn_regr = tf.zeros((output_height, output_width, num_anchors * 4))

  num_bboxes = len(img_data['bboxes'])

  num_anchors_for_bbox = tf.zeros(num_bboxes).as_type(tf.int32)
  # 4 stands for jy, ix, anchor_ratio_idx, anchor_size_idx
  best_anchor_for_bbox = -1 * tf.ones((num_bboxes, 4)).as_type(tf.int32)
  best_iou_for_bbox = tf.zeros(num_bboxes).as_type(tf.float32)
  best_x_for_bbox = tf.ones((num_bboxes, 4)).as_type(tf.int32)
  best_dx_for_bbox = tf.ones((num_bboxes, 4)).as_type(tf.float32)

  # ground truth boxes, resized
  gt = tf.zeros((num_bboxes, 4))
  for idx, bbox in enumerate(img_data['bboxes']):
    gt[idx, 0] = bbox['x1'] * (resized_width / width)
    gt[idx, 2] = bbox['x2'] * (resized_width / width)
    gt[idx, 1] = bbox['y1'] * (resized_height / height)
    gt[idx, 3] = bbox['y2'] * (resized_height / height)
  
  # rpn ground truth
  for anchor_size_idx in range(len(anchor_sizes)):
    for anchor_ratio_idx in range(n_anchratios):
      anchor_w = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
      anchor_h = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
      for ix in range(output_width):
        x1 = downscale * (ix + 0.5) - anchor_w / 2
        x2 = downscale * (ix + 0.5) + anchor_w / 2
        # ignore boxes that go across image boundaries	
        if x1 < 0 or x2 > resized_width:
          continue
        for jy in range(output_height):
          y1 = downscale * (jy + 0.5) - anchor_h / 2
          y2 = downscale * (jy + 0.5) + anchor_h / 2
          if y1 < 0 or y2 > resized_height:
            continue
          
          # whether an anchor should be a target
          bbox_type = 'neg'
          # this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
          best_iou_for_loc = 0.0

          for bbox_num in range(num_bboxes):
            curr_iou = iou([gt[bbox_num, 0], gt[bbox_num, 1], gt[bbox_num, 2], gt[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > config.rpn_max_overlap:
              cx_gt = (gt[bbox_num, 0] + gt[bbox_num, 2]) / 2
              cy_gt = (gt[bbox_num, 1] + gt[bbox_num, 3]) / 2
              cx = (x1 + x2) / 2
              cy = (y1 + y2) / 2
              tx = (cx_gt - cx) / (x2 - x1)
              ty = (cy_gt - cy) / (y2 - y1)
              tw = tf.math.log((gt[bbox_num, 2] - gt[bbox_num, 0]) / (x2 - x1))
              th = tf.math.log((gt[bbox_num, 3] - gt[bbox_num, 1]) / (y2 - y1))
            if img_data['bboxes'][bbox_num]['class'] != 'bg':
              # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
              if curr_iou > best_iou_for_bbox[bbox_num]:
                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                best_iou_for_bbox[bbox_num] = curr_iou
                best_x_for_bbox[bbox_num, :] = [x1, y1, x2, y2]
                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]
              # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
              if curr_iou > config.rpn_max_overlap:
                bbox_type = 'pos'
                num_anchors_for_bbox[bbox_num] += 1
                if curr_iou > best_iou_for_loc:
                  best_iou_for_loc = curr_iou
                  best_regr = (tx, ty, tw, th)
              # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
              if config.rpn_min_overlap < curr_iou < config.rpn_max_overlap:
                if bbox_type != 'pos':
                  bbox_type = 'neutral'
          if bbox_type == 'neg':
            y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
            y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
          elif bbox_type == 'neutral':
            y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
            y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
          elif bbox_type == 'pos':
            y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
            y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
            start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
            y_rpn_regr[jy, ix, start:start+4] = best_regr

  # ensure every bbox has at least one positive RPN region
  for idx in range(num_anchors_for_bbox.shape[0]):
    if num_anchors_for_bbox[idx] == 0:
      # no box with an IOU greater than zero ...
      if best_anchor_for_bbox[idx, 0] == -1:
        continue
      jy, ix, anchor_ratio_idx, anchor_size_idx = best_anchor_for_bbox[idx, 0]
      y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
      y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
      start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
      y_rpn_regr[jy, ix, start:start+4] = best_dx_for_bbox[idx, :]

  # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions, to make pos==neg. 
  # We also limit it to maximal 256 regions.
  num_regions = 256
  pos_locs = tf.where(y_is_box_valid == 1 & y_rpn_overlap == 1)
  neg_locs = tf.where(y_is_box_valid == 1 & y_rpn_overlap == 0)
  num_pos = len(pos_locs[0])
  num_neg = len(neg_locs[0])
  num_regions = 256
  
  if num_pos > num_regions/2:
    del_locs = random.sample(range(num_pos), num_pos - num_regions/2)
    y_is_box_valid[pos_locs[0][del_locs], pos_locs[1][del_locs], pos_locs[2][del_locs]] = 0
    y_rpn_overlap[pos_locs[0][del_locs], pos_locs[1][del_locs], pos_locs[2][del_locs]] = 0
    num_pos = num_regions/2
  
  if num_neg + num_pos > num_regions:
    del_locs = random.sample(range(num_neg), num_neg - num_pos)
    y_is_box_valid[pos_locs[0][del_locs], pos_locs[1][del_locs], pos_locs[2][del_locs]] = 0

  y_rpn_cls = tf.concat((y_is_box_valid, y_rpn_overlap), axis=-1)
  y_rpn_cls = tf.expand_dims(y_rpn_cls, axis=0)
  y_rpn_regr = tf.expand_dims(y_rpn_regr, axis=0)
  
  return y_rpn_cls, y_rpn_regr

# get ground truth anchor
def get_anchor_gt(img_data, class_count, config, img_length_calc_func):
  img_data_aug, x_img = data_augment(img_data, config)
  width, height =img_data_aug['width'], img_data_aug['height']
  rows, cols, _ = x_img.shape
  
  assert cols == width
  assert rows == height

  (resized_width, resized_height) = get_new_img_size(width, height, config.im_size)
  x_img = tf.image.resize_images(x_img, (resized_width, resized_height))
  y_rpn_cls, y_rpn_regr = cal_rpn(config, img_data_aug, width, height, resized_width, resized_height, img_length_calc_func)

  x_img[:, :, 0] -= config.img_channel_mean[0]
  x_img[:, :, 1] -= config.img_channel_mean[1]
  x_img[:, :, 2] -= config.img_channel_mean[2]
  x_img /= config.img_scaling_factor

  x_img = tf.expand_dims(x_img, axis=0)
  y_rpn_regr[:, :, :, y_rpn_regr.shape[-1]//2:] *= config.std_scaling

  return x_img, [y_rpn_cls, y_rpn_regr], img_data_aug
