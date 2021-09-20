import pandas as pd  # veri işleme, CSV dosyaları
import re  # Regex için gerekli modul
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = open("jokes.csv", "r", encoding="utf8")


# text = re.sub("'(\w+)", "", text)
def cleanData(data):
    punctuationSet = string.punctuation
    text = word_tokenize(data)
    data = data.lower()
    data = data.replace("\n ", " ")
    return data


#
# print(word_tokenize(text))
text = "".join(list(map(lambda x: x if x not in punctuationSet else " ", text)))

print(text)
# print(text)
