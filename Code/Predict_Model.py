#Chuẩn bị dữ liệu
import nltk
import Define
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD
import random
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import load_model
from math import sqrt
import keras
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

words = []
classes = []
documents = []
# Loại bỏ các từ / ký tự vô nghĩa
ignore_words = ['?', '!', '.', ',', 'a', 'v', 'xét']
# Đọc dữ liệu
data_file = open("data.json", encoding="utf8").read()
data_json = json.loads(data_file)
total = 0
for intent in data_json["intents"]:
    for pattern in intent["patterns"]:
        total = total + 1
        pattern = Define.no_accent_vietnamese(pattern).strip()
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# print(words)
words = [w for w in words if w not in ignore_words]
# print(words)
words = sorted(list(set(words)))
# print(words)
classes = sorted(list(set(classes)))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# #training
training = []
output_empty = [0]*len(classes)
# print(output_empty)
for doc in documents:
    bag = []

    pattern_words = doc[0]
    pattern_words = [word.lower() for word in pattern_words]
    # print(pattern_words)
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
train_x = np.array(train_x)
train_y = np.array(train_y)
y = []
for i in train_y:
    for j in range(0,len(i)):
        if i[j] == 1:
            y.append(j)
#print(y)

#Giải thuật SVM
train_x = np.array(train_x)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
clf = SVC(kernel='linear', C=1e5)
score_test = []
model = clf.fit(train_x, y)
for i in range(0, 10):
     X_train, X_test, y_train, y_test = train_test_split(train_x, y, test_size=1/3.0, random_state=100+i)
     keras.backend.clear_session()
     print("SVM accuracy: ", accuracy_score(y_test, clf.predict(X_test)))
print(score_test)
test = "ngành hệ thống thông tin xét học bạ sao ạ"
arr_test = Define.bow(test, words)
print(arr_test)
score = clf.predict([arr_test])
print(score[0])
print(classes[score[0]])
score = accuracy_score(y_test, clf.predict(X_test))
