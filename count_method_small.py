import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess, create_co_matrix, ppmi
from dataset import ptb

text = ' One of the several things that I do to stay healthy is to go jogging in my neighborhood. The reason why I do it is because it doesn’t require any special equipment. All I need to do is to put my shoes on and go outside, so it’s pretty easy to stick with. Another thing is to eat a healthy, well-balanced diet. Eating more vegetables and less salty food makes me feel well. People often say “you are what you eat,” and I think the saying is definitely true.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = create_co_matrix(corpus, vocab_size, window_size=15)
W = ppmi(C)

#SVDのやつ
U, S, V = np.linalg.svd(W)

print(C[0]) #共起行列

print(W[0]) #PPMI行列

print(U[0]) #SVD

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()
