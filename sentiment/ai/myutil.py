# from collections import defaultdict
import string
import numpy as np
from spacy.lang.vi import Vietnamese
from gensim.models.keyedvectors import KeyedVectors

def tokenize(texts):
    nlp = Vietnamese()
    docs = []
    for text in texts:
        tokens = np.array([postprocess_token(token.text) for token in nlp(text.lower())])
        docs.append(tokens)

    return docs

def postprocess_token(token):
    if token in string.punctuation:
        return '<punct>'
    elif token.isdigit():
        return '<number>'
    else:
        return token

nb_words = 10000
def missing_value():
    return nb_words

def make_embedding(texts, embedding_path, max_features=40000):
    embedding_path = str(embedding_path)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if embedding_path.endswith('.vec'):
        embedding_index = dict(get_coefs(*o.strip().split(" "))
                               for o in open(embedding_path))
        mean_embedding = np.mean(np.array(list(embedding_index.values())))
    elif embedding_path.endswith('bin'):
        embedding_index = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        mean_embedding = np.mean(embedding_index.vectors, axis=0)
    
    embed_size = mean_embedding.shape[0]
    word_index = sorted(list({word.lower() for sentence in texts for word in sentence}))
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    i = 1
    missing_words = []
    word_map = dict()
    for word in word_index:
        if i >= max_features:
            continue
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            missing_words.append(word)
            embedding_matrix[i] = mean_embedding
        word_map[word] = i
        i += 1
    
    embedding_matrix[-1] = mean_embedding
    np.savetxt("missing_words.csv", np.asarray(missing_words), fmt="%s", delimiter=",")
    return embed_size, word_map, embedding_matrix

def text_to_sequences(texts, word_map, max_len=100):
    texts_id = []
    max_nb_words = len(word_map)
    for sentence in texts:
        sentence = [word_map.get(word.lower(), max_nb_words) for word in sentence][:max_len]
        padded_setence = np.pad(sentence, (0, max(0, max_len - len(sentence))), 'constant', constant_values=0)
        texts_id.append(padded_setence)
    return np.array(texts_id)