# -*- coding: utf-8 -*-

import cPickle as pickle
import os

import ml.src.GlobalVariable

file_path = os.path.join(ml.src.GlobalVariable.datadir, "vocab.pkl")

data = pickle.load(open(file_path))

print data
