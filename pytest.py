#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pypinyin import pinyin, lazy_pinyin, Style
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi

txt = u'锄禾日当午'

py = lazy_pinyin(txt)

hmmparams = DefaultHmmParams()
result = viterbi(hmm_params=hmmparams, observations=py, path_num = 1)
for item in result:
  txt_rtn = u''.join(item.path)

if txt == txt_rtn:
  print u'OK'
else:
  print u'Error: %s -> %s -> %s' % (txt, py, txt_rtn)