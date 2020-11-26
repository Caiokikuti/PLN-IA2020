import re, code, string, unicodedata
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('stopwords')
nltk.download('wordnet')

def removeHtml(review):
    soup = BeautifulSoup(review, "html.parser")
    return soup.get_text()
	
def remove_between_square_brackets(review):
    return re.sub('\[[^]]*\]', '', review)

def remove_special_characters(review, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    review=re.sub(pattern,'',review)
    return review

def denoise_review(review):
    review = removeHtml(review)
    review = remove_between_square_brackets(review)
    review = remove_special_characters(review)
    return review

def remove_stopwords(tokens, language='english'):
    stopword = stopwords.words(language)
    text = [word for word in tokens if word not in stopword]
    return text

def lemmatizing(tokens):
    wn = WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokens]
    return text

# def classificacao(corpus, y):
#     divisao = 5
#     kfold = KFold(n_splits=divisao, shuffle=True, random_state=0)
#     y_t = []
#     y_p = []

#     for train_index, test_index in kfold.split(corpus):
#         x_train, x_test = corpus[train_index], corpus[test_index]
#         vectorizer = TfidfVectorizer(ngram_range=(1,1))

#         y_train, y_test = y[train_index], y[test_index]

#         clf = SVC(kernel='linear')
#         clf.fit(x_train, y_train.ravel())
#         y_pred = clf.predict(x_test)

#         y_t.extend(y_test)
#         y_p.extend(y_pred)

#     return y_t, y_p
def classificacao(corpus, y):
    split = 5
    kf = KFold(n_splits=split, shuffle=True, random_state=0)
    y_t = []
    y_p = []

    for train_index, test_index in kf.split(corpus):
        # print("TRAIN:", train_index, "TEST:", test_index)
        
        x_train, x_test = corpus[train_index], corpus[test_index]
        vectorizer = CountVectorizer(ngram_range=(1, 1))
        # vectorizer = CountVectorizer()
        
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        
        # print(x_train.shape, x_test.shape)

        y_train, y_test = y[train_index], y[test_index]

        # clf = SVC(kernel='linear')
        clf = LinearSVC()
        clf.fit(x_train, y_train.ravel())
        y_pred = clf.predict(x_test)

        y_t.extend(y_test)
        y_p.extend(y_pred)

    return y_t, y_p

def main():
    # Primeiro passo lendo o data set e colocando as labels
    dataSet = pd.read_csv('IMDB_Dataset.csv', encoding='latin-1')
    dataSet.dropna(how="any", inplace=True, axis=1)
    dataSet.colums = ['review', 'sentiment']

    dataSet.head()
    print(dataSet['sentiment'].value_counts())

    dataSet['labelNum'] = dataSet.sentiment.map({'negative':0,'positive':1})
    dataSet.head()
    # Processamento do texto
    reviews = dataSet['review']
    # tqdm.pandas()
    reviews = reviews.apply(lambda m: m.lower())
    # Retirando tags html e caracteres impuros
    reviews = reviews.apply(denoise_review)
    # Tokenizer
    reviews = reviews.apply(lambda m: m.split(' '))
    #  Removendo palavras vazias
    reviews = reviews.apply(remove_stopwords)
    # Normalizando
    reviews_lem = reviews.apply(lemmatizing)
    # Retornando para o texto
    reviews = reviews.apply(lambda t: ' '.join(t))
    reviews_lem = reviews_lem.apply(lambda t: ' '.join(t))
    # Escrevendo conjuntos no data set
    dataSet['reviews_sem_norm'] = reviews
    dataSet['reviews_lem'] = reviews_lem
    # testar e treinar 
    x_train, x_test, y_train, y_test = train_test_split(reviews_lem, dataSet['labelNum'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(ngram_range=(1,1), analyzer='word', max_features=100)
    vectorizer.fit(x_train)
    vectors = vectorizer.transform(x_train).todense()

    y_t, y_p = classificacao(dataSet['reviews_lem'], dataSet['labelNum'])

    print(f'F1-Score: {metrics.f1_score(y_t, y_p)}')
    
if __name__== "__main__":
    main()