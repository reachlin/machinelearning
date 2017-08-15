#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


font = ImageFont.truetype("data/SourceHanSerifSC-Regular.otf", 100)
str2draw = u"零一二三四五六七八九"
for char2draw in str2draw:
  img = Image.new( 'L', (128,128), "white")
  draw = ImageDraw.Draw(img)
  draw.text((0, 0), char2draw, align="center", font=font)
  img.save("data/%s.bmp" % hex(ord(char2draw)))