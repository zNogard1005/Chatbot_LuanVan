# Thêm câu hỏi ở ngoài vào

# Chuẩn bị nguyên liệu
import nltk
import Define
import json
import pickle
import numpy as np
import random


words = []
classes = []
documents = []

# Loại bỏ các từ / ký tự vô nghĩa
ignore_words = ['?', '!', '.', ',', 'a', 'v', 'xét']

# Đọc dữ liệu
data_file = open("data.json", encoding="utf8").read()
data_json = json.loads(data_file)
# print(data_json)

# Nhãn dữ liệu
def tag_data():
    tag = []
    for intent in data_json['intents']:
        tag.append(intent['tag'])
    return tag

# Hàm kiểm tra
def Test(question):
    question = Define.no_accent_vietnamese(question)
    tag = tag_data()
    for intent in data_json['intents']:
        for pattern in intent['patterns']:
            if Define.no_accent_vietnamese(pattern) == question:
                return 1, []
    return 0, tag

# Hàm thêm câu hỏi
def Add_question_intotag(question, tag):
    for intent in data_json['intents']:
        if intent["tag"] == tag:
            intent["patterns"].append(question)
    with open("data.json", "w", encoding='utf-8') as file:
        json.dump(data_json, file, ensure_ascii=False, indent=3)
    return "Đã thêm câu hỏi " + question + " vào nhãn " + tag

# Hàm thêm nhãn
def Add_new_tag(question, tag, ans):
    dict = {
        "tag": tag,                     # Nhãn mới
        "patterns": [question],         # Câu hỏi mới
        "responses": [ans]              # Câu trả lời
    }
    for intent in data_json['intents']:
        if intent["tag"] == tag:
            return "Nhãn đã tồn tại trong tập dữ liệu"
    data_json['intents'].append(dict)
    with open("data.json", "w", encoding='utf-8') as file:
        json.dump(data_json, file, ensure_ascii=False, indent=3)
    return "Đã thêm thành công"
# print(data_json["intents"])

# if __name__ == "__main__":
#     Test = Add_quesstion_intotag("diem chuan nganh Khoa hoc may tinh nam nay", "phuongthucxettuyen")
#     Test = Add_new_tag("diem chuan nganh Khoa hoc may tinh nam nay", "diemchuan", "diem chuan nam nay chua thong ke")
#     print(Test)
