#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

def get_img(path):
  im = Image.open(path)
  # convert to black and white
  im.convert('1')
  return list(im.getdata())

def get_data(wanted_labels):
  # 0, 1, 2, ..., 9
  chinese_num = ["0x96f6", "0x4e00", "0x4e8c", "0x4e09", "0x56db",
    "0x4e94", "0x516d", "0x4e03", "0x516b", "0x4e5d"]
  images = []
  labels = []
  for wanted in wanted_labels:
    images.append(get_img("data/%s.bmp" % chinese_num[wanted]))
    # label must be in one-hot format
    label = [0] * 10
    label[wanted] = 1
    labels.append(label)
  return images, labels

def get_train_data():
  return get_data([0,1,2,3,4,5,6,7,8,9])

def get_test_data():
  return get_data([0,4,3,7,0,0,1,2,5,9,8,8])