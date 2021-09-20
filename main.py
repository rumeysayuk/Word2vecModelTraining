from nltk.tokenize import word_tokenize, sent_tokenize
import re  # Regex için gerekli modul
import string
from nltk.corpus import stopwords

f = open("jokes.csv", "r", encoding="utf8")  # r =Read   utf8 turkçe karakterlerle çalışcağımızı belirttik.
data = f.read()


def clean(text):
    punctuation_set = string.punctuation
    ineffective_element_set = stopwords.words("english")
    text = sent_tokenize(text, "english")
    text = "".join(list(map(lambda x: x if x not in punctuation_set else " ", text)))
    text = re.sub("[0-9]+", "", text)
    text = text.lower()
    text = text.replace("\n ", " ")
    text = " ".join([i for i in text.split() if i not in ineffective_element_set])
    text = " ".join([i for i in text.split() if len(i) > 1])
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace("'", "")
    text = text.replace("``", "")
    return text


print(clean(data))
