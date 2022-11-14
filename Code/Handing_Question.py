# Tiền xử lý dữ liệu

# Chuẩn bị nguyên liệu
import re
import Define
from underthesea import sent_tokenize
tu_khoa = {"cong nghe thong tin chat luong cao": 0,
           "cong nghe thong tin": 0,
           "an toan thong tin": 1,
           "truyen thong da phuong tien": 2,
           "khoa hoc may tinh": 3,
           "ky thuat may tinh": 4,
           "mang may tinh": 5,
           "ky thuat phan mem": 6,
           "he thong thong tin": 7}
#Hàm xử lý dấu chấm câu
def Punctuation(string):
    punctuation = '''!()-[]{};:'"\<>./?@#$%^&*_~'''
    for x in string.lower():
        if x in punctuation:
            string = string.replace(x, "")
    # In chuỗi không có dấu chấm câu
    return string

#
def Handing(s):
    dict = {}
    s = Define.no_accent_vietnamese(s)
    result = []
    s = sent_tokenize(s)
    for k in s:
        k = Punctuation(k)
        if len(k) == 0:
            continue
        k = list(k.split())
        for i in range(len(k)):
            if k[i] == ('va' or '?'):
                k[i] = ','
        result = " ".join(k)
        result = result.split(',')
        # print(result)
        our_hello(result)
        result = Handing_1(result)
        for j in result:
            dict[j] = -1
            for h in tu_khoa:
                if h in j:
                    dict[j] = tu_khoa[h]
    return dict

#
def Handing_1(msg):
    chuoi = []
    for i in msg:
        s = i.strip()
        number = 1
        for k in tu_khoa:
            if number == 1:
                if k in s:
                    tmp = s.replace(k, "")
                    chuoi.append(tmp.strip())
                    chuoi.append(k)
                    number = 0
            else:
                break
        if number == 1:
            chuoi.append(s)
    # print("Chuoi: ", chuoi)
    for arr in chuoi:
        if arr == "":
            chuoi.remove(arr)
    result = []
    for i in range(len(chuoi)):
        if chuoi[i] in tu_khoa:
            for j in range(i, -1, -1):
                if chuoi[j] not in tu_khoa:
                    result.append(chuoi[j] + " " + chuoi[i])
                    break
        else:
            if i != len(chuoi) - 1:
                if chuoi[i+1] not in tu_khoa:
                    result.append(chuoi[i])
            else:
                result.append(chuoi[i])
    if len(result) == 0:
        return msg
    return result

#Hàm chào hỏi
def our_hello(res):
    hello = ["chao thay", "chao co", "cho em hoi", "cho hoi", "da", "thua thay", "thua co", ""]
    for k in hello:
        if k in res:
            res.remove(k)

#Kiểm tra
# if __name__ == "__main__":
#     msg = "ngành AN TOÀN THÔNG tin"
#     msg = Define.no_accent_vietnamese(msg)
#     print(msg)
#     k = Handing(msg)
#     print(k)
#     for i in k:
#         print(i)

#Kết quả
# nganh an toan thong tin
# ['nganh an toan thong tin']
# Chuoi:  ['nganh', 'an toan thong tin']
# {'nganh an toan thong tin': 1}
# nganh an toan thong tin
