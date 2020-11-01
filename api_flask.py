import pickle  
import pandas as pd 
import numpy as np 
from pyvi import ViTokenizer, ViPosTagger 
from flask import Flask, jsonify,request

#Load model ML
Pkl_Filename = "LR_Model.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file) 

#Load 
tfidf_path = "tfidf.pkl"
with open(tfidf_path, 'rb') as file:  
    tfidf_tokenizer = pickle.load(file) 

def predictor(text):
    tokenize = tfidf_tokenizer.transform([text])
    pred = Pickled_LR_Model.predict(tokenize)
    if pred == 0:
        return 'negative'
    elif pred == 1:
        return 'neutral'
    else :
        return 'positive'


def text_postag(text):
    pos_tag = ViPosTagger.postagging(ViTokenizer.tokenize(text))
    dict_tag = {}
    for i in range(len(pos_tag[0])):
        dict_tag[pos_tag[0][i]] = pos_tag[1][i]

    return dict_tag

def word_count(text_list):
    str = " ".join(text_list)
    str = ViTokenizer.tokenize(str)
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

def find_count(word,text_list,dict_tag):
    list_count_type = []
    for key,value in word_count(text_list).items():
        if key == word:
            list_count_type.append(value)
    for key,value in dict_tag.items():
        if key == word:
            list_count_type.append(value)
    return list_count_type 

def convert_json(line,text_list):
    dict_tag = text_postag(line)
    line = ViTokenizer.tokenize(line)
    list_json = [ ]
    for word in line.split():
        if word != None:
            list_json.append(
                {
                    'word' : word,
                    'type_word': find_count(word,text_list,dict_tag)[1],
                    'count_word' : find_count(word,text_list,dict_tag)[0]
                }
            )
    return list_json

comments = ['ba lô mỏng quá. shiper giao hàng nhanh.' , 'sản phẩm không có hướng dẫn rõ cho người dùng', ' Sản phẩm thì tốt nhưng shipper giao hàng cho tôi thì quá tệ.']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        total_message = []
        for comment in comments:
            total_message.append(
                {
                    'text': comment,
                    'prediction' : predictor(comment),
                    'information_word': convert_json(comment,comments)
                }
            )
        return jsonify(total_message)

if __name__ == '__main__':
   app.run(debug=True)






