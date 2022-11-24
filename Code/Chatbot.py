# Thư Viện
import Define
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request, redirect
from flask_ngrok import run_with_ngrok
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import Database
import Handing_Question
import Test_Question

Model = load_model("chatbot_model.h5")
intents = json.load(open("data.json", encoding="utf8"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def home():
    return render_template("Chatbot.html")

@app.route("/get1")
def chatbot_response():
    msg1 = request.args.get('msg')
    connection = Database.create_connection("sm_app.sqlite")
    msg = Handing_Question.Handing(msg1)
    print(msg)
    res = ""
    # t="test2"
    tag_temp = ""
    for i in msg:
        # print(i)
        int = predict_class(i, Model)
        # print(msg[i])
        r, tag_question = getResponse(int, intents, msg[i])
        tag_temp = tag_temp + tag_question + " "
        res = res + "</br>" + r + "</br>"
        print(res)
    mp3, len = fileaudio_text(tag_temp)
    Database.execute_query_insert(connection, msg1, tag_temp)
    return res + " " + mp3 + str(len)

@app.route("/sql")
def my_sql():
    connection = Database.create_connection("sm_app.sqlite")
    select_users = "SELECT * from users"
    rows = Database.execute_read_query(connection, select_users)
    return render_template("Index_B1812282.html", rows=rows)

@app.route('/delete/<int:id>', methods=['GET', 'POST'])
def delete_sql(id):
    connection = Database.create_connection("sm_app.sqlite")
    Database.execute_query_delete(connection, id)
    return redirect("/sql")

@app.route("/get2")
def fileaudio_text(tag):
    tag = tag.strip()
    file = tag.split(" ")
    print(file)
    r = ""
    for i in file:
        r = r + "static/audio/"+i+".mp3"+" "
    return r, len(file)

@app.route('/add_question/<string:str>',methods=['GET','POST'])
def add_ques(str):
    text = str
    x, tag = Test_Question.Test(str)
    if x == 1:
        results = "Câu hỏi đã tồn tại trong cơ sở dữ liệu huấn luyện"
        return render_template("add.html", results=results, key=x)
    else:
        return render_template("add.html", results=tag, key=x, qes=text)

@app.route('/process_add',methods=['GET','POST'])
def process_add():
    if request.method == 'POST':
        question = request.form["cauhoi"]
        tag = request.form["tag"]
        note = Test_Question.Add_question_intotag(question, tag)
        return render_template("add.html", res=note)

@app.route('/process_add_tag',methods=['GET','POST'])
def process_add_tag():
    if request.method == 'POST':
        question = request.form["cauhoi"]
        tag = request.form["tag"]
        ans = request.form["cautraloi"]
        note = Test_Question.Add_new_tag(question, tag, ans)
        return render_template("add.html", res1=note)

def bow(sentence, words):
    # tokenize the pattern
    sentence_words = Define.clean_up_sentence(sentence)
    # print(sentence_words)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, Model):
    p = bow(sentence, words)
    temp = np.array([0]*len(p))
    if np.array_equal(p, temp):
        return []
    res = Model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.8
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({"intents": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intent_json, stt):
    if len(ints) == 0:
        return "Xin lỗi, hiện tại câu hỏi này bot chưa thể giải đáp, bot sẽ ghi nhận câu hỏi và cải thiện chất lượng", "no_answer"
    print(ints)
    #print(intent_json)
    print(stt)
    tag = ints[0]["intents"]
    list_of_intents = intent_json["intents"]
    result = ""
    res_tag = ""
    for i in list_of_intents:
        # if i["tag"] != tag:
        #     continue
        #
        # result = i['responses'][0]
        # res_tag = tag
        # break
        if stt == -1:
            if i["tag"] == tag:
                if len(i['responses']) == 1:
                    result = i['responses'][0]
                    res_tag = tag
                    break
                else:
                    result = i['responses'][8]
                    res_tag = tag+str(8)
                    break
        else:
            if i["tag"] == tag:
                if len(i['responses']) == 1:
                    result = i['responses'][0]
                    res_tag = tag
                    break
                else:
                    result = i['responses'][stt]
                    res_tag = tag + str(stt)
                    break

    return result, res_tag

if __name__ == "__main__":
    app.run()