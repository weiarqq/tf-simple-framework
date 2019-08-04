import numpy as np
import tensorflow as tf



def load_glove_embeddings(path, word_index, vocab_size, embedding_size):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            w = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embeddings[w] = vectors

    embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
    num_loaded = 0
    for w, i in word_index.items():
        v = embeddings.get(w)
        if v is not None and i < vocab_size:
            embedding_matrix[i] = v
            num_loaded += 1
    print('Successfully loaded pretrained embeddings for {}/{} words.'.format(num_loaded, vocab_size))
    embedding_matrix = embedding_matrix.astype(np.float32)
    return embedding_matrix


def load_data(path='imdb.npz',
              num_words=None,
              skip_top=0,
              maxlen=None,
              seed=113,
              start_char=1,
              oov_char=2,
              index_from=3,
              **kwargs):

      # Legacy support

      with np.load(path,allow_pickle=True) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

      np.random.seed(seed)
      indices = np.arange(len(x_train))
      np.random.shuffle(indices)
      x_train = x_train[indices]
      labels_train = labels_train[indices]

      indices = np.arange(len(x_test))
      np.random.shuffle(indices)
      x_test = x_test[indices]
      labels_test = labels_test[indices]

      xs = np.concatenate([x_train, x_test])
      labels = np.concatenate([labels_train, labels_test])

      if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
      elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

      if not num_words:
        num_words = max([max(x) for x in xs])

      # by convention, use 2 as OOV word
      # reserve 'index_from' (=3 by default) characters:
      # 0 (padding), 1 (start), 2 (OOV)
      if oov_char is not None:
        xs = [
            [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
        ]
      else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

      idx = len(x_train)
      x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
      x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

      return (x_train, y_train), (x_test, y_test)



