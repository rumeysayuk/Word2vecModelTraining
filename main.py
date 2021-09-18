import nltk
from nltk.corpus import stopwords
import pandas as pd
import string
import re

data = pd.read_csv("jokes.csv")
news_dataset = data[["Answer"]]


# print(news_dataset.head(3))
def cleanData(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub("'(\w+)", "", text)
    
