#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

def get_img(path):
  im = Image.open(path)
  im.thumbnail((128,128), Image.ANTIALIAS)
  # convert to black and white
  im.convert('1')
  return list(im.getdata())

def get_data(wanted_labels, folders=['data'], onehot=True):
  # 0, 1, 2, ..., 9
  chinese_num = ["0x96f6", "0x4e00", "0x4e8c", "0x4e09", "0x56db",
    "0x4e94", "0x516d", "0x4e03", "0x516b", "0x4e5d"]
  images = []
  labels = []
  for wanted in wanted_labels:
    for folder in folders:
      images.append(get_img("%s/%s.bmp" % (folder, chinese_num[wanted])))
      if onehot:
        # label must be in one-hot format
        label = [0] * 10
        label[wanted] = 1
        labels.append(label)
      else:
        labels.append(wanted)
  return images, labels

def get_train_data(folders=["data", "data/hard1", "data/hard2"], onehot=True):
  return get_data([0,1,2,3,4,5,6,7,8,9], folders, onehot)

def get_test_data(folders=["data/light"], onehot=True):
  return get_data([0,4,3,7,0,0,1,2,5,9,8,8], folders, onehot)