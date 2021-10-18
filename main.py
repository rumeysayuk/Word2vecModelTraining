import numpy as np
import string
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# f = open("turkish_news_70000.csv/turkish_news_70000.csv", "r",
#          encoding="utf8")
# text = f.read()
#
#
# def cleandata(data):
#     punctuation_set = string.punctuation
#     ineffective_element_set = stopwords.words("turkish")
#     ineffective_element_set.extend(
#         ["www", "jpg", "wp", "com", "php", "id", "aa", "src", "nin", "mi", "dha", "nı", "ni", "tr", "li", "ın", "rde",
#          "ün", "un", "nun", "mızın", "tan", "ta", "te", "nın", "ye", "la", "https", "in", "göre", "olsa", "ler",
#          "leri", "son", "na", "http", "co", "sb", "sn", "ila", "bin", "nbir", "bir", "iki", "üç", "content", "text",
#          "main", "image", "published", " site", " text", "title", "url", "td", "ı", "i", "png", "ilk", "capital",
#          "birinci", "ikinci", "ucuncu","lik","lık","uncu","üncü","e","a"])
#     data = "".join(list(map(lambda x: x if x not in punctuation_set else " ", data)))
#     data = re.sub("[0-9]+", "", data)
#     data = data.lower()
#     data = data.replace("\n ", " ")
#     data = " ".join([i for i in data.split() if i not in ineffective_element_set])
#     data = " ".join([i for i in data.split() if len(i) > 1])
#     data = re.sub(r'[^\w\s]', '', data)
#     data = data.replace("'", "")
#     data = data.replace("``", "")
#     return data
#
#
# newData = cleandata(text)
# f = open("cleanDatas.txt", "w", encoding="utf8")
# f.write(newData)
# f = open("cleanDatas.txt", "r", encoding="utf8")
# text = f.read()
f = open("cleanDatas.txt", "r", encoding="utf8")  # r =Read   utf8 turkçe karakterlerle çalışcağımızı belirttik.
text = f.read()
t_list = text.split('\n')

corpus = []
for sentence in t_list:
    a = corpus.append(sentence.split())
# print(corpus[:10])

model = Word2Vec(corpus, vector_size=100, window=5, min_count=5, sg=1)
# print(model)
# sg=1 skip-gram alg kullanılacak demek.default cbow kullanılıyor

# print(model.wv["ankara"])

# print(model.wv.most_similar("istanbul"))
# modele kelimeleri verip eğittik.Ülkeleri bağdaştırıp bize getirdi.

# print(model.wv.most_similar("pazartesi"))
# gün ile ilgili bağdaştırdıklarını getiriyor.

# model.save("word2vec.model")
# model = Word2Vec.load("word2vec.model")
#
#
def closeswords_tsneplot(model, word):
    word_vectors = np.empty((0, 100))
    word_labels = [word]

    close_words = model.wv.most_similar(word)
    word_vectors = np.append(word_vectors, np.array([model.wv[word]]), axis=0)
    # axis=0 verilmezse vektörler düzleştirilir.

    for w, _ in close_words:
        word_labels.append(w)
        word_vectors = np.append(word_vectors, np.array([model.wv[w]]), axis=0)
    tsne = TSNE(random_state=0)
    y = tsne.fit_transform(word_vectors)

    x_coords = y[:, 0]
    y_coords = y[:, 1]

    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(5, -2), textcoords="offset points")
    plt.show()


closeswords_tsneplot(model, "erdoğan")
