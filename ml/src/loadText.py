# -*- coding: utf-8 -*-

import cPickle
import codecs
import collections
import os

import numpy as np

import GlobalVariable


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input3.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)

        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data) #Make Dictionary Key : unicode character, Value : # of Key
        count_pairs = sorted(counter.items(), key=lambda x: -x[1]) #Sort Descending
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f) # Dump characters
        self.tensor = np.array(list(map(self.vocab.get, data))) # Get으로 key의 value를 가지고온다. 인풋 파일을 키의 밸류로 바꾼다?
        np.save(tensor_file, self.tensor)
        '''
        counter : text 파일 내에 있는 모든 유니코드 글자로 딕셔너리를 만든다. 키는 글자, value는 나온 횟수
        count_pairs : counter의 내용을 리스트로 만듦.
        self.chars : text 파일 내에 있는 모든 유니코드 글자
        self.vocab_size : self.chars의 개수.
        self.vocab : 중요함  -  많이 나온 순서대로 딕셔너리를 만든다. 키는 글자, value는 많이 나온 순서.
        '''

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tesor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0



a = TextLoader(GlobalVariable.datadir, 1, 1);
