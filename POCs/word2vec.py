import gensim
from gensim.models import Word2Vec

# model = gensim.models.Word2Vec(corpus,
#                                min_count = 1, size = 100, window = 5)
from  gensim.models import KeyedVectors
# from  gensim.models.KeyedVectors import load_word2vec_format
# '../input/word2vec-google/GoogleNews-vectors-negative300.bin',
# './word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin',
# './GoogleNews-vectors-negative300.bin',

def load_word2vec():
    word2vecDict = KeyedVectors.load_word2vec_format(
        './GoogleNews-vectors-negative300.bin.gz',
        binary=True, unicode_errors='ignore')
    embeddings_index = dict()
    for word in word2vecDict.wv.vocab:
        embeddings_index[word] = word2vecDict.word_vec(word)

    return embeddings_index

def get_word_embeddings(word_list):
    embeddings_list = []
    word2vecDict = KeyedVectors.load_word2vec_format(
        './GoogleNews-vectors-negative300.bin.gz',
        binary=True, unicode_errors='ignore')

    for word in word_list:
        embeddings_list.append(word2vecDict.word_vec(word))

    return embeddings_list

if __name__ == "__main__":
    # w2v_model=load_word2vec()
    # print("w2v_model['London'].shape")
    # print(w2v_model['London'].shape)
    # print(w2v_model['London'])
    print(get_word_embeddings(['London','tea','coffee']))
