import sys
sys.path.append('..')
import numpy as np
from preprocess import most_similar, create_co_matrix,ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
W = ppmi(C, verbose=True)

print('calculating SVD ...')
try:
    from sklean.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

except ImportError:
    U, S, V = np.linalg.svd(W)

    word_vecs = U[:, :wordvec_size]

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

