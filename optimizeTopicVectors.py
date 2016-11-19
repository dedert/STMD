import numpy as np

# vector initialize
def orthogonal_matrix(shape):
    # Stolen from blocks:
    # github.com/mila-udem/blocks/blob/master/blocks/initialization.py
    M1 = np.random.randn(shape[0], shape[0])
    M2 = np.random.randn(shape[1], shape[1])

    # QR decomposition of matrix with entries in N(0, 1) is random
    Q1, R1 = np.linalg.qr(M1)
    Q2, R2 = np.linalg.qr(M2)
    # Correct that NumPy doesn't force diagonal of R to be non-negative
    Q1 = Q1 * np.sign(np.diag(R1))
    Q2 = Q2 * np.sign(np.diag(R2))

    n_min = min(shape[0], shape[1])
    return np.dot(Q1[:, :n_min], Q2[:n_min, :])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def loss(topicVec, n_wkl, W, lamda = 0.01):
    score = np.dot(W, topicVec)
    regular = lamda * np.linalg.norm(topicVec, ord=2)
    word_count = np.sum(n_wkl,(1,2)).reshape(1,-1)
    loss = - np.dot(word_count, (score - np.log(np.exp(score).sum()))) + regular
    return loss

def grad(topicVec, n_wkl, W, lamda = 0.01):
    word_count = np.sum(n_wkl, (1,2)).reshape(1,-1)
    score = np.dot(W, topicVec)
    first_factor = np.dot(word_count, W - (softmax(score).reshape(-1,1) * W)) #(1,100)
    regular = 2 * lamda * topicVec #(100,1)
    grad = - first_factor.reshape(-1,1) + regular.reshape(-1,1)
    return np.squeeze(grad)
