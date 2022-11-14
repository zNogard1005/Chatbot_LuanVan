#Tiền xử lý dữ liệu

# Chuẩn bị nguyên liệu
import re
from scipy import spatial
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import math

# Danh sách từ vô nghĩa
list_word_nomean = ['ạ', 'đi', 'em', 'thầy', 'ơi', 'e', 'cô', 'vậy', 'à']
# Hàm xóa từ vô nghĩa
def delete_word(s):
    #Loại bỏ các ký tự kéo dài VD: ơnnnnnnnnn
    s = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), s, flags=re.IGNORECASE)
    #Chuyển thành chữ thường
    s = s.lower()
    s = s.split()
    for i in range(len(s)):
        if s[i] in list_word_nomean:
            s[i] = ""
    return " ".join(s)
# Hàm xóa dấu câu
def no_accent_vietnamese(s):
    s = delete_word(s)
    s = abbreviation(s)
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s

# Hàm xử lý chữ viết tắt
def abbreviation(s):
    s = s.replace('cnt', 'cong nghe thong tin')
    s = s.replace('at', 'an toan thong tin')
    s = s.replace('tdpt', 'truyen thong da phuong tien')
    s = s.replace('khmt', 'khoa hoc may tinh')
    s = s.replace('ktmt', 'ky thuat may tinh')
    s = s.replace('mt', 'mang may tinh')
    s = s.replace('ktpm', 'ky thuat phan mem')
    s = s.replace('ht', 'he thong thong tin')
    s = s.replace('clc', 'chat luong cao')
    # s = s.replace('ttdl', 'truyen thong du lieu')
    s = s.replace('bn', 'bao nhieu')
    s = s.replace('ko', 'khong')
    s = s.replace('ntn', 'nhu the nao')
    s = s.replace('ktx', 'ky tuc xa')
    s = s.replace('nhiu', 'nhieu')
    s = s.replace('hc', 'hoc')
    s = s.replace('dc', 'duoc')
    s = s.replace('nghanh', 'nganh')
    return s

# Hàm xử lý khoảng cách hàng
def distance_row(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i]-row2[i])**2
    return math.sqrt(distance)

#Hàm làm sạch câu
def clean_up_sentence(sentence):
    # sentence = sentence.lover()
    sentence_words = no_accent_vietnamese(sentence)
    # sentence_words= alias(sentence_words)
    # print(sentence_words)
    sentence_words = nltk.word_tokenize(sentence_words)
    sentence_words = [word.lower() for word in sentence_words]
    # print(sentence_words)
    return sentence_words

#Hàm bow
def bow(sentence, words):
    # Mã hóa mẫu
    sentence_words = clean_up_sentence(sentence)
    # túi từ - ma trận N từ, ma trận từ vựng
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
            # Thực hiện gán giá trị 1, nếu vị trí của từ hiện tại ở vị trí từ vựng
                bag[i] = 1
    # print(np.array(bag))
    return (np.array(bag))

#Hàm tính độ chính xác
def accuracy_cosine(predict, test_Y):
    count = 0
    for i in range(0, len(predict)):
        if predict[i] == test_Y[i]:
            count += 1
    return (count/len(predict))*100




