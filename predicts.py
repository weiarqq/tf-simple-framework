import os
import string
import tempfile
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from tensorboard import summary as summary_lib
from input_functions import train_input_fn, eval_input_fn
from utils import load_glove_embeddings, load_data
from models import cnn_model_fn, lstm_model_fn

vocab_size = 5000
sentence_size = 200
embedding_size = 50
index_offset = 3
project_path = os.path.dirname(os.path.abspath(__file__))
model_dir = 'model'
word_index = imdb.get_word_index(os.path.join(project_path, 'data/imdb_word_index.json'))


embedding_matrix = load_glove_embeddings('data/glove.6B.50d.txt', word_index, vocab_size, embedding_size)
params = {'embedding_initializer': embedding_matrix}

lstm_classifier = tf.estimator.Estimator(model_fn=lstm_model_fn,
                                                   model_dir=os.path.join(model_dir, 'cnn_pretrained'),
                                                   params=params)

def print_predictions(sentences):
    def text_to_index(sentence):
        # Remove punctuation characters except for the apostrophe
        translator = str.maketrans('', '', string.punctuation.replace("'", ''))
        tokens = sentence.translate(translator).lower().split()
        return np.array([1] + [word_index[t] + index_offset if t in word_index else 2 for t in tokens])

    indexes = [text_to_index(sentence) for sentence in sentences]
    x = sequence.pad_sequences(indexes,
                               maxlen=sentence_size,
                               padding='post',
                               value=0)
    label = np.array([min(len(x), sentence_size) for x in indexes])
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x, "len": label}, shuffle=False)
    labels = lstm_classifier.predict(input_fn=predict_input_fn)
    pr = [p['logistic'] for p in labels]
    print(pr)
    # for idx, sentence in enumerate(sentences):
    #     print(sentence)
    #     print(str(predictions[idx]))



print_predictions([
    'I really liked the movie!',
    'Hated every second of it...'])