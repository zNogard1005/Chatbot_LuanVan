# Chuẩn bị nguyên liệu
from pyvi import ViTokenizer, ViPosTagger
from pyvi import ViUtils
import nltk
import Define
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import keras
from keras.models import load_model

if __name__ == "__main__":
    words = []
    classes = []
    documents = []
    #Từ lược bỏ
    ignore_words = ['?', '!', '.', ',', 'a', 'v', 'xét']
    data_file = open('data.json').read()
    intents = json.loads(data_file)
    # k = ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội, ngành công nghệ thông tin")
    # k1 = ViPosTagger.postagging(ViTokenizer.tokenize("Trường đại học bách khoa hà nội, ngành công nghệ thông tin"))
    # print(k1)
    # k2 = ViUtils.remove_accents("")
    # print(k2.decode('utf-8'))