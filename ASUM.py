import numpy as np
import pandas as pd
from collections import Counter
from scipy.optimize import fmin_l_bfgs_b
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


class ASUM:
    def __init__(self, pos_seed, neg_seed, numTopics, alpha, beta, gamma, max_sentence=50, numSentiments=2):
        self.pos_seed = pos_seed
        self.neg_seed = neg_seed
        self.numTopics = numTopics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numSentiments = numSentiments
        self.maxSentence = max_sentence

    def build_dataset(self, reviews, sentiment_list):
        """
        :param reviews: 리뷰 데이터 [ [[문서1의 문장1],[문서1의 문장2]], [[문서2의 문장1],[문서2의 문장2]], ...]]
        :return:
        """
        corpus = [word for review in reviews for sentence in review for word in sentence]
        text = nltk.Text(corpus)
        freq = nltk.FreqDist(text)
        keywords = [tup[0] for tup in freq.most_common(len(freq.keys()))]  # 많이 등장한 단어순으로 index
        word2idx = {}  # key : 단어, value : index
        for index, key in enumerate(keywords):
            word2idx[key] = index

        idx2word = dict(zip(word2idx.values(), word2idx.keys()))  # key : index, value : 단어
        doc_sent_word_dict = {}  # key: 문서 index, value : [[list of sent1 단어의 index], [list of sent2 단어의 index]...]
        numSentence = {}  # key : 문서 index, value : 해당 문서의 문장수
        wordCountSentence = {}  # key : 문서 index, value : 해당 문서의 각 문장별 word count
        docSentiment = {}
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
            docSentiment[index] = sentiment_list[index]

        return word2idx, idx2word, doc_sent_word_dict, wordCountSentence, numSentence, docSentiment

    def _initialize_(self, reviews, pos_neg_sentence_indices, pos_neg_sentiment_label, sentiment_list):
        self.word2idx, self.idx2word, self.doc_sent_word_dict, self.wordCountSentence, \
        self.numSentence, self.docSentiment = self.build_dataset(reviews, sentiment_list)
        self.numDocs = len(self.doc_sent_word_dict.keys())
        self.vocabSize = len(self.word2idx.keys())
        self.pos_neg_sentence_indices = pos_neg_sentence_indices
        self.pos_neg_sentiment_label = pos_neg_sentiment_label

        # Pseudocounts
        self.n_wkl = np.zeros((self.vocabSize, self.numTopics, self.numSentiments))  # 단어 i가 topic k, senti l로 할당된 수
        self.n_kl = np.zeros((self.numTopics, self.numSentiments))  # topic k, senti l로 할당된 단어 수
        self.ns_d = np.zeros((self.numDocs))  # 문서 d의 문장 수
        self.ns_dkl = np.zeros((self.numDocs, self.numTopics, self.numSentiments))  # 문서 d에서 topic k, sentiment l로 할당된 문장 수
        self.ns_dl = np.zeros((self.numDocs, self.numSentiments))  # 문서 d에서 sentiment l로 할당된 문장 수
        self.topics = {}
        self.sentiments = {}

        alphaVec = self.alpha * np.ones(self.numTopics)
        gammaVec = self.gamma * np.ones(self.numSentiments)

        for d in range(self.numDocs):
            topicDistribution = sampleFromDirichlet(alphaVec)
            sentimentDistribution = np.zeros((self.numTopics, self.numSentiments))

            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)

            for m in range(self.numSentence[d]):
                t = sampleFromCategorical(topicDistribution)
                count = 0
                pos_senti = 0
                neg_senti = 0
                for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):
                    if w in self.pos_seed:
                        pos_senti += 1
                        count += 1
                    elif w in self.neg_seed:
                        neg_senti += 1
                        count += 1

                if count == 0:
                    s = sampleFromCategorical(sentimentDistribution[t, :])
                else:
                    if pos_senti == neg_senti:
                        s = sampleFromCategorical(sentimentDistribution[t, :])
                    elif pos_senti > neg_senti:
                        s = 0
                    else:
                        s = 1

                self.topics[(d, m)] = t  # d 문서의 m번째 문장의 topic
                self.sentiments[(d, m)] = s  # d 문서의 m 번째 문장의 sentiment
                self.ns_d[d] += 1
                self.ns_dkl[d, t, s] += 1
                self.ns_dl[d, s] += 1
                for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):  # d번째 문서의 m번째 문장의 단어를 돌면서
                    self.n_wkl[w, t, s] += 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
                    self.n_kl[t, s] += 1  # topic k, senti l로 할당된 단어 수


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
                       (self.ns_dl[d] + self.numTopics * self.alpha)[np.newaxis,:] # dim(L x 1)

        thirdFactor = (self.ns_dl[d] + self.gamma) / \
                      (self.ns_d[d] + self.numSentiments * self.gamma)

        prob = np.ones((self.numTopics, self.numSentiments))
        prob *= firstFactor * secondFactor
        prob *= thirdFactor[np.newaxis,:]
        prob /= np.sum(prob)

        ind = sampleFromCategorical(prob.flatten())
        t, s = np.unravel_index(ind, prob.shape)

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

    def perplexity(self):
        """
        exp(-1 * log-likelihood per word)
        log-likelihood =
        :return:
        """

    def getTopKWordsByLikelihood(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(t, s | v) is maximum
        """
        pseudocounts = np.copy(self.n_wkl)
        normalizer = np.sum(pseudocounts, (1, 2))
        pseudocounts /= normalizer[:, np.newaxis, np.newaxis]
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                print(t, s, [self.idx2word[i] for i in topWordIndices])

    def getTopKWordsByTS(self, K):
        """
        K 개 sentiment별 top words
        """
        topic_sentiment_arr = self.calculatePhi()
        dic = {}
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                index_list = np.argsort(-topic_sentiment_arr[:, t, s])[:K]
                if s == 0:
                    name = "p"
                else:
                    name = "n"
                dic['topic_' + '{:02d}'.format(t + 1) + '_' + name] = [self.idx2word[index] for index in index_list]
        return pd.DataFrame(dic)

    def getTopKWordsByTopic(self, K):
        dic = {}
        phi = self.calculatePhi()
        topic_arr = np.sum(phi, (2))
        for t in range(self.numTopics):
            index_list = np.argsort(-topic_arr[:, t])[:K]
            dic["Topic"+str(t+1)] = [self.idx2word[index] for index in index_list]
        return pd.DataFrame(dic)

    def getTopicSentimentDist(self, d):
        theta = self.calculateTheta()[d]
        return theta

    def getDocSentimentDist(self, d):
        pi = self.calculatePi()[d]
        return pi

    def getTopWordsBySenti(self, K):
        dic = {}
        phi = self.calculatePhi()
        senti_arr = np.sum(phi, (1))
        for s in range(self.numSentiments):
            index_list = np.argsort(-senti_arr[:, s])[:K]
            if s == 0:
                name = "p"
            else:
                name = "n"
            dic["Sentiment_"+ name] = [self.idx2word[index] for index in index_list]
        return pd.DataFrame(dic)

    def classify_senti(self):
        doc_sent_inference = []
        for i in range(self.numDocs):
            if i in self.pos_neg_sentence_indices:
                doc_sent_inference.append(np.argmax(self.getDocSentimentDist(i)))
        infer_arr = np.array(doc_sent_inference)
        answer = np.array(self.pos_neg_sentiment_label)
        return np.mean(infer_arr == answer)

    def save(self, iteration, path):
        phi = self.calculatePhi()
        theta = self.calculateTheta()
        pi = self.calculatePi()
        name = path + "_topic_" + '{:03d}'.format(self.numTopics) + '_iter_' + str(iteration+1)
        np.save(name + "_phi", phi)
        np.save(name + "_theta", theta)
        np.save(name + "_pi", pi)

    def run(self, reviews, save_path, print_iter=2, save_iter = 5, maxIters=10):
        for iteration in range(maxIters):
            if (iteration + 1) % print_iter == 0:
                print("Starting iteration %d of %d" % (iteration + 1, maxIters))
                print(self.classify_senti())
            if (iteration + 1) % save_iter == 0:
                print("Starting save model")
                self.save(iteration, save_path)

            for d in range(self.numDocs):
                for m in range(self.numSentence[d]):
                    self.sampling(d, m)
