from nltk.tokenize import word_tokenize, sent_tokenize
import re  # Regex için gerekli modul
import string
import pandas as pd
from nltk.corpus import stopwords

f = open("jokes.csv", "r", encoding="utf8")  # r =Read   utf8 turkçe karakterlerle çalışcağımızı belirttik.
text = f.read()
# text = sent_tokenize(text)
text = word_tokenize(text)

punctuationSet = string.punctuation
text = "".join(list(map(lambda x: x if x not in punctuationSet else " ", text)))
text = re.sub("[0-9]+", "", text)
text = text.lower()
text = text.replace("\n ", " ")
ineffectiveElementSet = stopwords.words("english")
text = " ".join([i for i in text.split() if i not in ineffectiveElementSet])
text = " ".join([i for i in text.split() if len(i) > 1])
text = re.sub("'(\w+)", "", text)
print(text)
