# Chuẩn bị nguyên liệu
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import keras
import Define
import random
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
import copy


def process_data() -> tuple:
    lemmatizer = WordNetLemmatizer()


    # Khởi tạo dữ liệu
    words = []
    classes = []
    documents = []
    # Loại bỏ các từ / ký tự vô nghĩa
    ignore_words = ['?', '!', '.', ',', 'a', 'v', 'xét']
    # Đọc dữ liệu
    data_file = open("data.json", encoding="utf8").read()
    data_json = json.loads(data_file)
    # print(data_json)

    # Xử lý dữ liệu
    for intent in data_json["intents"]:
        for pattern in intent["patterns"]:
            pattern = pattern.lower()
            pattern = Define.no_accent_vietnamese(pattern).strip()

            # Mã hóa từng từ
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))

            # Thêm vào danh sách nhãn
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

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
    output_empty = [0] * len(classes)
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
    training = np.array(training)
    np.random.shuffle(training)
    # print(training)
    # Tạo tập huấn luyện và tập test. X - patterns, Y - intents
    train_x = np.array([np.array(ele) for ele in training[:, 0]])
    train_y = np.array([np.array(ele) for ele in training[:, 1]])
    print("Tập dữ liệu được khởi tạo xong")

    return train_x, train_y

def main() -> None:
    X, Y = process_data()
    # one hot to label
    Y = np.argmax(Y, axis=1)
    # Sua model
    model = GaussianNB()

    # k-fold
    avg_acc = 0
    cnt = 0
    # n_splits la so lan lap
    kf = KFold(n_splits=10, shuffle=False)
    for train_idx, test_idx in kf.split(X, Y):
        cnt = cnt + 1

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        clf = copy.deepcopy(model)
        clf.fit(X_train, Y_train)

        Y_pred = clf.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print(f"Độ chính xác lần lặp thứ {cnt} là: {acc}")

        avg_acc = avg_acc + acc

    avg_acc = avg_acc / kf.get_n_splits()
    print(f"Độ chính xác trung bình sau {kf.get_n_splits()}-fold là: {avg_acc}")

    # hold out
    for i in range(0, 10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=2/3.)

        clf = copy.deepcopy(model)
        clf.fit(X_train, Y_train)

        Y_pred = clf.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        avg_acc = avg_acc + acc
        print(f"Độ chính xác của nghi thức Hold-out lần lặp thứ {i} là: {acc}")
    avg_acc = avg_acc / 10
    print("Độ chính xác sau 10 lần lặp của nghi thức Hold-out là: ", avg_acc)

if __name__ == "__main__":
    main()