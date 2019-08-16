#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#Get dataset from local directory
df = pd.read_csv('D:/Programacion/Datasets/IMDB/IMDB_Dataset.csv', encoding='utf-8')
print(df.head())

#Set the dataset labels to have a numerical value
le = LabelEncoder()
labels = le.fit_transform(df['sentiment'].values)
x_train, x_test, y_train, y_test = train_test_split(df['review'].values, labels, test_size=0.2)

#Tokenize text and transform it into tf-idf encoding
def tokenizer(text):
    return text.split()

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        tokenizer=tokenizer)

x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

#Define, fit and evaluate classifier
clf = LogisticRegression(C=10.0, penalty='l2')
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
