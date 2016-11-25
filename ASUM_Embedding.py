import numpy as np
import time
from collections import Counter
import optimizeTopicVectors as ot
from preprocess import *

def sampleFromDirichlet(alpha):
    return np.random.dirichlet(alpha)


def sampleFromCategorical(theta):
    # theta = theta / np.sum(theta)
    return np.random.multinomial(1, theta).argmax()


def word_indices(doc_sent_word_dict, sent_index):
    """
    :param doc_sent_word_dict:
    :param sent_index:
    :return:
    """
    sentence = doc_sent_word_dict[sent_index]
    for idx in sentence:
        yield idx

class ASUM_Embedding:
    def __init__(self, review_label, wordVectors, sentimentVector, numTopics, alpha, beta, gamma, binary=0.5, numSentiments=2):
        self.review_label = review_label # 각 문서의 긍정(0), 부정(1) label
        self.wordVectors = wordVectors # (V x H)
        self.numTopics = numTopics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numSentiments = numSentiments
        self.binary = binary
        self.sentimentVector = sentimentVector # (L x H)

    def build_dataset(self, reviews):
        """
        :param reviews: 리뷰 데이터 [ [[문서1의 문장1],[문서1의 문장2]], [[문서2의 문장1],[문서2의 문장2]], ...]]
        :return:
        word2idx - key: word, value: index
        idx2word - key: index, value: word
        doc_sent_word_dict -
        """
        corpus = [word for review in reviews for sentence in review for word in sentence]
        text = nltk.Text(corpus)
        freq = nltk.FreqDist(text)
        keywords = [tup[0] for tup in freq.most_common(self.wordVectors.shape[0])]  # 많이 등장한 단어 선택
        word2idx = {}  # key : 단어, value : index
        for index, key in enumerate(keywords):
            word2idx[key] = index

        idx2word = dict(zip(word2idx.values(), word2idx.keys()))  # key : index, value : 단어
        doc_sent_word_dict = {}  # key: 문서 index, value : [[list of sent1 단어의 index], [list of sent2 단어의 index]...]
        numSentence = {}  # key : 문서 index, value : 해당 문서의 문장수
        wordCountSentence = {}  # key : 문서 index, value : 해당 문서의 각 문장별 word count
        for index, review in enumerate(reviews):
            doc_sent_lst = []
            doc_sent_count = []
            for sent in review:
                word_indices = [word2idx[word] for word in sent if word in word2idx]
                doc_sent_lst.append(word_indices)
                counts = Counter(word_indices)
                doc_sent_count.append(counts)
            numSentence[index] = len(doc_sent_lst)
            doc_sent_word_dict[index] = doc_sent_lst
            wordCountSentence[index] = doc_sent_count

        return word2idx, idx2word, doc_sent_word_dict, wordCountSentence, numSentence

    def _initialize_(self, reviews):
        self.word2idx, self.idx2word, self.doc_sent_word_dict, self.wordCountSentence, \
        self.numSentence = self.build_dataset(reviews)
        self.numDocs = len(self.doc_sent_word_dict.keys())
        self.vocabSize = len(self.word2idx.keys())

        # Pseudocounts
        self.n_wkl = np.zeros((self.vocabSize, self.numTopics, self.numSentiments))  # 단어 i가 topic k, senti l로 할당된 수
        self.n_kl = np.zeros((self.numTopics, self.numSentiments))  # topic k, senti l로 할당된 단어 수
        self.ns_d = np.zeros((self.numDocs))  # 문서 d의 문장 수
        self.ns_dkl = np.zeros((self.numDocs, self.numTopics, self.numSentiments))  # 문서 d에서 topic k, sentiment l로 할당된 문장 수
        self.ns_dl = np.zeros((self.numDocs, self.numSentiments))  # 문서 d에서 topic k로 할당된 문장 수
        self.topics = {}
        self.sentiments = {}
        self.senti_score_dict = {}

        alphaVec = self.alpha * np.ones(self.numTopics)
        gammaVec = self.gamma * np.ones(self.numSentiments)

        for d in range(self.numDocs):
            topicDistribution = sampleFromDirichlet(alphaVec)
            sentimentDistribution = np.zeros((self.numTopics, self.numSentiments))

            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)

            for m in range(self.numSentence[d]):
                t = sampleFromCategorical(topicDistribution)
                pos_score = np.dot(self.sentimentVector,
                                   self.wordVectors[self.doc_sent_word_dict[d][m]].T).sum(axis=1)
                s = np.argmax(pos_score)
                self.senti_score_dict[(d, m)] = ot.softmax(pos_score)
                self.topics[(d, m)] = t  # d 문서의 m번째 문장의 topic
                self.sentiments[(d, m)] = s  # d 문서의 m 번째 문장의 sentiment
                self.ns_d[d] += 1
                self.ns_dkl[d, t, s] += 1
                self.ns_dl[d, s] += 1
                for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):  # d번째 문서의 m번째 문장의 단어를 돌면서
                    self.n_wkl[w, t, s] += 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
                    self.n_kl[t, s] += 1  # topic k, senti l로 할당된 단어 수



    def inference(self, d, m):
        t = self.topics[(d, m)]
        s = self.sentiments[(d, m)]
        self.ns_d[d] -= 1
        self.ns_dkl[d, t, s] -= 1
        self.ns_dl[d, s] -= 1
        for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):
            self.n_wkl[w, t, s] -= 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
            self.n_kl[t, s] -= 1  # topic k, senti l로 할당된 단어 수

        firstFactor = np.ones((self.numTopics, self.numSentiments))

        word_count = self.wordCountSentence[d][m]
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                beta0 = self.n_kl[t][s] + self.beta
                m0 = 0
                for word in word_count.keys():
                    betaw = self.n_wkl[word, t, s] + self.beta
                    cnt = word_count[word]
                    for i in range(cnt):
                        firstFactor[t][s] *= (betaw + i) / (beta0 + m0)
                        m0 += 1

        secondFactor = (self.ns_dkl[d, :, :] + self.alpha) / \
                       (self.ns_dl[d] + self.numTopics * self.alpha)[np.newaxis, :]  # dim(K x L)

        thirdFactor = (self.ns_dl[d] + self.gamma) / \
                      (self.ns_d[d] + self.numSentiments * self.gamma)  # (L,)

        prob = np.ones((self.numTopics, self.numSentiments))
        prob *= firstFactor * secondFactor
        prob *= thirdFactor[np.newaxis, :]
        prob /= np.sum(prob)

        #원래대로 돌려놓음
        self.topics[(d, m)] = t
        self.sentiments[(d, m)] = s
        self.ns_d[d] += 1
        self.ns_dkl[d, t, s] += 1
        self.ns_dl[d, s] += 1
        for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):
            self.n_wkl[w, t, s] += 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
            self.n_kl[t, s] += 1  # topic k, senti l로 할당된 단어 수

        ind = sampleFromCategorical(prob.flatten())
        t, s = np.unravel_index(ind, prob.shape)

        senti_score = self.senti_score_dict[(d, m)]
        s2 = sampleFromCategorical(senti_score.flatten())
        gamma = np.random.binomial(1, self.binary) #binary가 0이면 항상 0만 추출
        s = (1 - gamma) * s + gamma * s2
        return t, s


    def sampling(self, d, m):
        t = self.topics[(d, m)]
        s = self.sentiments[(d, m)]
        self.ns_d[d] -= 1
        self.ns_dkl[d, t, s] -= 1
        self.ns_dl[d, s] -= 1
        for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):
            self.n_wkl[w, t, s] -= 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
            self.n_kl[t, s] -= 1  # topic k, senti l로 할당된 단어 수

        firstFactor = np.ones((self.numTopics, self.numSentiments))

        word_count = self.wordCountSentence[d][m]
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                beta0 = self.n_kl[t][s] + self.beta
                m0 = 0
                for word in word_count.keys():
                    betaw = self.n_wkl[word, t, s] + self.beta
                    cnt = word_count[word]
                    for i in range(cnt):
                        firstFactor[t][s] *= (betaw + i) / (beta0 + m0)
                        m0 += 1

        secondFactor = (self.ns_dkl[d, :, :] + self.alpha) / \
                       (self.ns_dl[d] + self.numTopics * self.alpha)[np.newaxis,:]  # dim(K x L)

        thirdFactor = (self.ns_dl[d] + self.gamma) / \
                      (self.ns_d[d] + self.numSentiments * self.gamma) #(L,)

        prob = np.ones((self.numTopics, self.numSentiments))
        prob *= firstFactor * secondFactor
        prob *= thirdFactor[np.newaxis,:]
        prob /= np.sum(prob)

        ind = sampleFromCategorical(prob.flatten())
        t, s = np.unravel_index(ind, prob.shape)

        senti_score = self.senti_score_dict[(d, m)]
        s2 = sampleFromCategorical(senti_score.flatten())
        gamma = np.random.binomial(1, self.binary)
        s = (1-gamma) * s + gamma * s2
        self.topics[(d, m)] = t
        self.sentiments[(d, m)] = s
        self.ns_d[d] += 1
        self.ns_dkl[d, t, s] += 1
        self.ns_dl[d, s] += 1
        for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):
            self.n_wkl[w, t, s] += 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
            self.n_kl[t, s] += 1  # topic k, senti l로 할당된 단어 수

    def calculatePhi(self):
        firstFactor = (self.n_wkl + self.beta) / \
                      np.expand_dims(self.n_kl + self.n_wkl.shape[0] * self.beta, axis=0)
        firstFactor /= firstFactor.sum()
        return firstFactor

    def calculateTheta(self):
        secondFactor = (self.ns_dkl + self.alpha) / \
                       np.expand_dims(self.ns_dl + self.numTopics * self.alpha, axis=1)  # dim(K x 1)
        secondFactor /= secondFactor.sum()
        return secondFactor

    def calculatePi(self):
        thirdFactor = (self.ns_dl + self.gamma) / \
                      np.expand_dims(self.ns_d + self.numSentiments * self.gamma, axis=1)
        thirdFactor /= thirdFactor.sum()
        return thirdFactor


    def getTopicSentimentDist(self, d):
        theta = self.calculateTheta()[d]
        return theta

    def getDocSentimentDist(self, d):
        pi = self.calculatePi()[d]
        pi /= pi.sum()
        # doc_sent_word_dict = self.doc_sent_word_dict
        # wordInDocIndex = [index for index_list in doc_sent_word_dict[d] for index in index_list]
        # wordVector = self.wordVectors[wordInDocIndex] # num words in Sentence x dimension
        # senti_score = np.dot(self.sentimentVector, wordVector.T).sum(axis=1) #(2,)
        # pi = (1-self.binary) * pi + self.binary * ot.softmax(senti_score)
        # pi /= pi.sum()
        return pi

    def classify_senti(self):
        inference = np.argmax(self.calculatePi(), axis=1)  # (D x L)
        answer = np.array(self.review_label)
        return np.mean(inference == answer)

    def save(self, iteration, path):
        phi = self.calculatePhi()
        theta = self.calculateTheta()
        pi = self.calculatePi()
        name = path + "_topic_" + '{:03d}'.format(self.numTopics) + '_iter_' + str(iteration+1)
        np.save(name + "_phi", phi)
        np.save(name + "_theta", theta)
        np.save(name + "_pi", pi)

    def run(self, save_path, print_iter=2, save_iter = 5, maxIters=10):
        for iteration in range(maxIters):
            start = time.time()
            if (iteration + 1) % print_iter == 0:
                print("Starting iteration %d of %d:" % (iteration + 1, maxIters))
                print(self.classify_senti())
            if (iteration + 1) % save_iter == 0:
                print("Starting save model")
                self.save(iteration, save_path)

            for d in range(self.numDocs):
                for m in range(self.numSentence[d]):
                    self.sampling(d, m)
            end = time.time()
            print("iteration %s, time %i"%(iteration+1,end-start))
    # def updateTopicVectors(self, lamda = 0.01):
    #     t = self.topicVectors # (K, H)
    #     for i in range(self.numTopics):
    #         x0 = t[i, :]
    #         x, f, d = fmin_l_bfgs_b(ot.loss, x0, fprime=ot.grad, args=(self.n_wkl, self.wordVectors, lamda), maxiter=15000)
    #         t[i, :] = x
    #     self.topicVectors = t


    # topic_similarity = ot.softmax(np.dot(self.topicVectors,
    #                                      self.wordVectors[
    #                                          self.doc_sent_word_dict[d][m]].T))  # ( K x num words in sentence)
    # senti_similarity = ot.softmax(np.dot(self.sentimentVector,
    #                                      self.wordVectors[
    #                                          self.doc_sent_word_dict[d][m]].T))  # ( L x num words in sentence)
    # vector_similarity = ot.softmax(np.dot(topic_similarity, senti_similarity.T))

    # senti_similarity = np.dot(self.sentimentVector,
    #                           self.wordVectors[self.doc_sent_word_dict[d][m]].T).sum(axis=1)  # ( L, )