#!/usr/bin/env python
# coding: utf-8

# In[15]:


import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger','ner'])
nlp.max_len = len(content)*2


# In[29]:


import string  

puncs = string.punctuation+'\n\n\n'+' \n' + ' ” “' 
puncs


# In[11]:


def file_reader(file_path):
    with open(file_path,encoding="utf8") as f:
        content = f.read()
    return content


# In[12]:


content = file_reader('carol-alice.txt')


# In[13]:


content


# In[26]:


def punc_cleaner(content):
    return [token.text.lower() for token in nlp(content) if token.text not in puncs]


# In[30]:


clean_content = punc_cleaner(content)


# In[31]:


clean_content


# In[32]:


train_seq_size = 26
token_seq = []
for i in range(train_seq_size, len(clean_content)):
    seq = clean_content[i-train_seq_size:i]
    print(seq)
    token_seq.append(seq)


# In[45]:


from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


# In[38]:


keras_tokenizer = Tokenizer()
keras_tokenizer.fit_on_texts(token_seq)


# In[39]:


sequences = keras_tokenizer.texts_to_sequences(token_seq)


# In[42]:


keras_tokenizer.word_counts


# In[44]:


sample = sequences[0]
for i in sample:
    print(keras_tokenizer.index_word[i])


# In[47]:


seq_array = np.array(sequences)
seq_array.shape


# In[52]:


from tensorflow.keras.utils import to_categorical


# In[48]:


X = seq_array[:, :-1]
y = seq_array[:,-1]
X.shape


# In[49]:


vocab_size = len(keras_tokenizer.word_counts)


# In[53]:


seq_len = X.shape[1]
y_cat = to_categorical(y, num_classes=vocab_size+1)
y_cat[0]


# In[55]:


from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.models import Sequential


# In[56]:


def model_creator(vocab_size, seq_len):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=seq_len, input_length=seq_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    return model


# In[58]:


model = model_creator(vocab_size+1, seq_len)


# In[61]:


model.fit(X, y_cat, batch_size=128, epochs=100,verbose=1)


# In[62]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[110]:


def generate_text(model, keras_tokenizer, seed_text, pred_len):
    output_text = []
    input_text = seed_text
    for i in range(pred_len):
        sequence = keras_tokenizer.texts_to_sequences([input_text])[0]
        padded_sequence = pad_sequences([sequence], maxlen = 25,truncating='pre')
        pred = model.predict(padded_sequence)
        pred = np.argmax(pred)
#         print(pred)
#         print('!!')
        pred_word = keras_tokenizer.index_word[pred]
#         print(input_text+ ' '+ pred_word)
        input_text+= ' ' + pred_word
        output_text.append(pred_word)
    print(' '.join(output_text))


# In[111]:


sample = 'be no mistake about it: it was neither more nor less than a pig, and she felt that it would be quit'
generate_text(model, keras_tokenizer,sample,50)


# In[112]:


model.save('Alice_LSTM.h5')


# In[113]:


from pickle import dump
dump(keras_tokenizer, open('Alice_LSTM_Tokenizer', 'wb'))


# In[ ]:




