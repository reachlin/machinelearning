#!/usr/bin/env python
# -*- coding: utf-8 -*-


with open('mapping.txt', 'r') as m:
  mf = m.read().decode("UTF-8").split(';')
  mapping = {}
  for item in mf:
    line = item.split(',')
    mapping[line[0]] = line[1]
  with open('output.txt', 'r') as f:
    py = f.read().decode("UTF-8")

    sentence = []
    for itempy in py.split(' '):
      if itempy in mapping:
        sentence.append(mapping[itempy])
      else:
        sentence.append(itempy)
    print unicode(u' '.join(sentence)).encode('utf8')