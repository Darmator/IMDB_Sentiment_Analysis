#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Bidirectional, Dense, Embedding, CuDNNLSTM, Dropout, GlobalMaxPool1D

#Get dataset from local directory
df = pd.read_csv('D:/Programacion/Datasets/IMDB/IMDB_Dataset.csv', encoding='utf-8')
print(df.head())

#Set the dataset labels to have a numerical value
le = LabelEncoder()
labels = le.fit_transform(df['sentiment'].values)
x_train, x_test, y_train, y_test = train_test_split(df['review'].values, labels, test_size=0.2)

#Hyperparameters
max_features = 6000 #n_words in the embedding dictionary
max_words = 130 #n_words per review
input_dim = max_features #embedding layer input
embedding_dim = 128 #embedding layer output
epochs = 3
batch_size = 100

#Keras tokenizer
tokenizer_obj = Tokenizer(num_words = max_features)
tokenizer_obj.fit_on_texts(x_train)

#Fit tokenizer and turn texts into sequences of numbers
x_train = tokenizer_obj.texts_to_sequences(x_train)
x_test = tokenizer_obj.texts_to_sequences(x_test)
x_train = pad_sequences(x_train, maxlen=max_words)
x_test = pad_sequences(x_test, maxlen=max_words)

#Define model
model = Sequential()
model.add(Embedding(input_dim, embedding_dim))
model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(1,activation='sigmoid'))

#Compile
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
print(model.summary())

#Fit and evaluate
model.fit(x_train, y_train, batch_size=batch_size, epochs = epochs, validation_data=(x_test, y_test))