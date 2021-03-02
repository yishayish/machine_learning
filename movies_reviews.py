import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPool1D, Dropout


r=[]
s=[]
with open(r'C:\Users\User\Desktop\rt-polarity.neg',encoding='utf8',errors='ignore') as neg:
    for sentence in neg.readlines():
        r.append(sentence.strip())
        s.append(0)
with open(r'C:\Users\User\Desktop\rt-polarity.pos',encoding='utf8',errors='ignore') as pos:
    for sentence in pos.readlines():
        r.append(sentence.strip())
        s.append(1)
z=[*zip(r,s)]


df = pd.DataFrame(z,columns=['review','label'])


n=len(df)
train=int(n*0.8)

df = df.sample(frac = 1).reset_index(drop=True)

tokenizer=Tokenizer(num_words=6000)

tokenizer.fit_on_texts(df['review'])

text_sequence=tokenizer.texts_to_sequences(df.review)

pad_sequence=pad_sequences(text_sequence)

r_train, s_train, r_test, s_test= pad_sequence[:train], df.label[:train],pad_sequence[train:], df.label[train:]



model = Sequential()

model.add(Embedding(input_dim=6000, output_dim=64))
model.add(LSTM(32, return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 7
validation_split = 0.2
model.fit(x=r_train, y=s_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

pred = model.predict(x=r_test)
y_pred = (pred >= 0.5) * 1

correct=0
false=0
for i ,j in zip(s_test,y_pred):
    if i== j[0]:
        correct+=1
    else:
        false+=1
print(correct, false)
