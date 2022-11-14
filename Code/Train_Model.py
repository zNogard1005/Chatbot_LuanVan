# Huấn luyện mô hình - Giải Thuật SGD

# Chuẩn bị nguyên liệu
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import Define
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from keras import backend
import random
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import keras
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#Import và load tập dữ liệu
words = []
classes = []
documents = []
# Loại bỏ các từ / ký tự vô nghĩa
ignore_words = ['?', '!', '.', ',', 'a', 'v', 'xét']
# Đọc dữ liệu
data_file = open("data.json", encoding="utf8").read()
data_json = json.loads(data_file)
total = 0
# Tiền xử lý dữ liệu
for intent in data_json["intents"]:
    for pattern in intent["patterns"]:
        total = total + 1
        pattern = Define.no_accent_vietnamese(pattern).strip()

        # Mã hóa từng từ
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        # Thêm vào danh sách nhãn
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# print(total)
# print(documents)

# Chuyển thành chữ thường và xóa các từ trùng lặp
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sắp xếp nhãn
classes = sorted(list(set(classes)))
# documents = patterns + intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = từ, từ vựng
print(len(words), "unique lemmatized words", words)
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Huấn luyện
# Tạo dữ liệu huấn luyện
training = []
# Tạo mảng trống cho đầu ra
output_empty = [0]*len(classes)
# # Nhãn [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# # print(output_empty)
# Huấn luyện, túi từ cho mỗi câu
for doc in documents:
    # Khởi tạo cho túi từ
    bag = []
    # Danh sách các từ được mã hóa
    pattern_words = doc[0]
    # lemmatize từng từ - tạo từ cơ bản, cố gắng biểu diễn các từ liên quan
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Tách từ trong câu hỏi
    # print(pattern_words)
    # tạo mảng túi từ của chúng tôi với 1, nếu tìm thấy kết hợp từ trong mẫu hiện tại
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # đầu ra là '0' cho mỗi thẻ và '1' cho thẻ hiện tại (cho mỗi mẫu)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Chuyển về dạng np.array
random.shuffle(training)
training = np.array(training)
# print(training)
# Tạo tập huấn luyện và tập test. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Tập dữ liệu được khởi tạo xong")


# print(np.array(train_x))

activation = ['relu', 'sigmoid', 'tanh']


def create_model():
    model = Sequential()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, sgd


def get_activation_model(activation, train_x, train_y, X_test, Y_test):
    for k in activation:
        model, sgd = create_model()
        # print("Hàm kích hoạt: ", k)
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation=k))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation=k))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(np.array(train_x), np.array(train_y), epochs=80, batch_size=4, verbose=1)
        score = model.evaluate(np.array(X_test), np.array(Y_test))
        print("Độ chính xác của mô hình SGD với hàm kích hoạt ", k, " là: ", score[1])

print(len(words))


train_x = np.array(train_x)
train_y = np.array(train_y)

def model_test(train_x,train_y):
    model, sgd = create_model()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    # early_stopping_monitor = EarlyStopping(patience=2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=4, verbose=1)
    model.save('chatbot_model.h5', hist)
    return model
# #
# #
from sklearn.model_selection import train_test_split

score_test = []
for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=2/3., random_state=100+i)
#   # keras.backend.clear_session()
    model = model_test(X_train, y_train)
    score = model.evaluate(np.array(X_test), np.array(y_test))
    score_test.append(score[1])
    print("Độ chính xác của mô hình SGD với lần lặp thứ ", i, " là: ", score[1])
# #
print(score_test)
print(classes)

# # Đánh giá mô hình _ với tối ưu hóa SGD
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=10)
for train_index, test_index in ss.split(train_x):
    keras.backend.clear_session()
    # print("Train: ",train_index,"Test",test_index)
    X_test, X_train = train_x[test_index], train_x[train_index]
    Y_test, Y_train = train_y[test_index], train_y[train_index]
    get_activation_model(activation, X_train, Y_train, X_test, Y_test)
    model, sgd = create_model()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    # early_stopping_monitor = EarlyStopping(monitor='loss', patience=5, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    hist = model.fit(np.array(X_train), np.array(Y_train), epochs=100, batch_size=4, verbose=1)
    plt.plot(hist.history['accuracy'])
    # plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()
    plt.figure()
# # # #
# # # #
ss1 = ShuffleSplit(n_splits=3, test_size=0.4, random_state=13)
for train_index, test_index in ss1.split(train_x):
    # keras.backend.clear_session()
    # print("Train: ", train_index, "Test", test_index)
    X_test, X_train = train_x[test_index], train_x[train_index]
    Y_test, Y_train = train_y[test_index], train_y[train_index]
    score = model.evaluate(np.array(X_test), np.array(Y_test), verbose=0)
    print("accuracy_SGD: ", score[1])
# Lấy hàm "tanh" đưa ra giá trị tốt nhất
