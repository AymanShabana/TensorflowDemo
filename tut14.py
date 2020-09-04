import csv
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


vocab_size = 10000# YOUR CODE HERE
embedding_dim = 16# YOUR CODE HERE
max_length = 120# YOUR CODE HERE
trunc_type ='post' # YOUR CODE HERE
padding_type ='post' # YOUR CODE HERE
oov_tok ="<OOV>" # YOUR CODE HERE
training_portion = .8
num_epochs = 30

sentences = []
labels = []
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
print(len(stopwords))

with open("tmp2/bbc-text.csv", 'r') as csvfile:
    i = 0
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if i == 0:
            i = 1
            continue
        labels.append(row[0])
        tempSentence = ''
        wordList = re.sub("[^\w]", " ", row[1]).split()
        for word in wordList:
            if not word in stopwords:
                tempSentence += word + ' '
        sentences.append(tempSentence)

print(len(labels))
print(len(sentences))
print(sentences[0])

train_size = int(training_portion*len(labels)) # YOUR CODE HERE

train_sentences = np.array(sentences[0:train_size])# YOUR CODE HERE
train_labels = np.array(labels[0:train_size])# YOUR CODE HERE

validation_sentences = np.array(sentences[train_size:]) # YOUR CODE HERE
validation_labels =np.array(labels[train_size:]) # YOUR CODE HERE

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

validation_sequences = tokenizer.texts_to_sequences(validation_sentences)# YOUR CODE HERE
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)# YOUR CODE HERE

print(len(validation_sequences))
print(validation_padded.shape)

label_tokenizer = Tokenizer(num_words=vocab_size)# YOUR CODE HERE
label_tokenizer.fit_on_texts(labels)# YOUR CODE HERE)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))# YOUR CODE HERE
validation_label_seq =np.array(label_tokenizer.texts_to_sequences(validation_labels))# YOUR CODE HERE

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(len(training_label_seq))

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(len(validation_label_seq))

# Expected output
# [4]
# [2]
# [1]
# (1780, 1)
# [5]
# [4]
# [3]
# (445, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

# Expected output
# (1000, 16)