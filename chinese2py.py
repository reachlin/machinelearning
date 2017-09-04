#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pypinyin import pinyin, lazy_pinyin, Style

with open('input_chinese.txt', 'r') as f:
  chinese = f.read().decode("UTF-8")
  sentence = []
  mapping = []
  for char in chinese:
    py = lazy_pinyin(char)
    if py[0]==char:
      sentence.append(',')
    else:
      sentence.append(py[0])
      mapping.append("%s,%s" % (py[0], char))
  print ' '.join(sentence)
  print unicode(';'.join(mapping)).encode('utf8')
