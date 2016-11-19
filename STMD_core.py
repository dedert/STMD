import numpy as np
import nltk
import pandas as pd
from ast import literal_eval
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


class STMD_Gibbs_Sampler:
    def __init__(self, wordVectors, sentimentVector, numTopics, alpha, beta, gamma, binary=0.5, max_sentence=50, numSentiments=2):
        self.wordVectors = wordVectors # (V x H)
        self.numTopics = numTopics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numSentiments = numSentiments
        #self.MAX_VOCAB_SIZE = max_vocab_size
        self.maxSentence = max_sentence
        self.dimension = self.wordVectors.shape[1]  # H
        self.binary = binary
        self.sentimentVector = sentimentVector # (L x H)

    def build_dataset(self, reviews, sentiment_list):
        """
        :param reviews: 리뷰 데이터 [ [[문서1의 문장1],[문서1의 문장2]], [[문서2의 문장1],[문서2의 문장2]], ...]]
        :return:
        """
        corpus = [word for review in reviews for sentence in review for word in sentence]
        text = nltk.Text(corpus)
        freq = nltk.FreqDist(text)
        #keywords = [tup[0] for tup in freq.most_common(self.MAX_VOCAB_SIZE)]  # 많이 등장한 단어 선택
        keywords = [tup[0] for tup in freq.most_common(self.wordVectors.shape[0])]  # 많이 등장한 단어 선택
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
        self.topicVectors = ot.orthogonal_matrix((self.numTopics, self.dimension))

        # Pseudocounts
        self.n_wkl = np.zeros((self.vocabSize, self.numTopics, self.numSentiments))  # 단어 i가 topic k, senti l로 할당된 수
        self.n_kl = np.zeros((self.numTopics, self.numSentiments))  # topic k, senti l로 할당된 단어 수
        self.ns_d = np.zeros((self.numDocs))  # 문서 d의 문장 수
        self.ns_dkl = np.zeros((self.numDocs, self.numTopics, self.numSentiments))  # 문서 d에서 topic k, sentiment l로 할당된 문장 수
        self.ns_dk = np.zeros((self.numDocs, self.numTopics))  # 문서 d에서 topic k로 할당된 문장 수
        self.topics = {}
        self.sentiments = {}
        # self.priorSentiment = {}

        alphaVec = self.alpha * np.ones(self.numTopics)
        gammaVec = self.gamma * np.ones(self.numSentiments)

        for d in range(self.numDocs):
            topicDistribution = sampleFromDirichlet(alphaVec)
            sentimentDistribution = np.zeros((self.numTopics, self.numSentiments))

            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)

            for m in range(self.numSentence[d]):
                t = sampleFromCategorical(topicDistribution)
                # s = sampleFromCategorical(sentimentDistribution[t, :])
                s = self.docSentiment[d]
                self.topics[(d, m)] = t  # d 문서의 m번째 문장의 topic
                self.sentiments[(d, m)] = s  # d 문서의 m 번째 문장의 sentiment
                self.ns_d[d] += 1
                self.ns_dkl[d, t, s] += 1
                self.ns_dk[d, t] += 1
                for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):  # d번째 문서의 m번째 문장의 단어를 돌면서
                    self.n_wkl[w, t, s] += 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
                    self.n_kl[t, s] += 1  # topic k, senti l로 할당된 단어 수

    def updateTopicVectors(self, lamda = 0.01):
        t = self.topicVectors # (K, H)
        for i in range(self.numTopics):
            x0 = t[i, :]
            x, f, d = fmin_l_bfgs_b(ot.loss, x0, fprime=ot.grad, args=(sampler.n_wkl, wordVectors, lamda), maxiter=15000)
            t[i, :] = x
        self.topicVectors = t


    def conditionalDistribution(self, d, m):
        """
        Calculates the (topic, sentiment) probability for sentence m in document d
        Returns:    a matrix (numTopics x numSentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.numTopics, self.numSentiments))

        # firstfactor 수정
        prob = 1
        for word_idx in sampler.doc_sent_word_dict[d][m]:
            for i in range(sampler.wordCountSentence[d][m][word_idx]):
                prob *= sampler.n_wkl[word_idx, :, :] + sampler.beta + sampler.wordCountSentence[d][m][word_idx] - 1 - i
                prob /= prob.sum()

        firstFactor = prob # dim(K x L)
        # firstFactor1 = (self.n_wkl[w, :, :] + self.beta) / \
        #               (self.n_kl + self.n_wkl.shape[0] * self.beta)  # dim(K x L)

        # wordSoftmax = ot.softmax(np.dot(self.topicVectors, sampler.wordVectors.T))
        # sentiSoftmax = ot.softmax(np.dot(sentimentVector, sampler.wordVectors.T))
        # firstFactor2 = np.dot(wordSoftmax, sentiSoftmax.T)
        # firstFactor2 /= firstFactor2.sum() # (K x L)
        #
        # firstFactor = (self.binary * firstFactor1) * ((1-self.binary) * firstFactor2)
        # firstFactor /= np.sum(firstFactor)

        secondFactor = (self.ns_dk[d, :] + self.alpha) / \
                       (self.ns_d[d] + self.numTopics * self.alpha)  # dim(K x 1)

        thirdFactor = (self.ns_dkl[d, :, :] + self.gamma) / \
                      (self.ns_dk[d] + self.numSentiments * self.gamma)[:, np.newaxis]  # dim (K x L)

        probabilities_ts *= firstFactor * thirdFactor
        probabilities_ts *= secondFactor[:, np.newaxis]
        probabilities_ts /= np.sum(probabilities_ts)
        return probabilities_ts

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
                # vocab = self.vectorizer.get_feature_names()
                print(t, s, [self.idx2word[i] for i in topWordIndices])

    def getTopKWordsSentiments(self, K):
        """
        K 개 sentiment별 top words
        """
        lst = []
        normalizer = np.sum(pseudocounts, (1))
        word_prob = normalizer / np.sum(normalizer, 0)
        for s in range(self.numSentiments):
            topWordIndices = word_prob[:, s].argsort()[::-1][:K]
            lst.append([s, [self.idx2word[i] for i in topWordIndices]])
        print(lst)

    def getTopKWords(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(v | t, s) is maximum
        """
        topic_name = ['topic_' + str(i + 1) for i in range(sampler.numTopics)]
        sentiment_name = ['neg', 'pos']  # 1 이 긍정
        pseudocounts = np.copy(self.n_wkl)
        normalizer = np.sum(pseudocounts, (0))
        pseudocounts /= normalizer[np.newaxis, :, :]
        df = pd.DataFrame()
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                df['topic_' + str(t + 1) + '_' + sentiment_name[s]] = [self.idx2word[i] for i in topWordIndices]
        return df

    def getTopic(self, d):
        topic_name = ['topic_' + str(i + 1) for i in range(sampler.numTopics)]
        df = pd.DataFrame(columns=topic_name)
        theta_d = (self.ns_dk[d, :] + self.alpha) / \
                  (self.ns_d[d] + self.numTopics * self.alpha)  # dim(K x 1)
        df.loc[len(df)] = theta_d
        return df

    def getDocSentiment(self, d):
        theta_d = (self.ns_dk[d, :] + self.alpha) / \
                  (self.ns_d[d] + self.numTopics * self.alpha)  # dim(K x 1)
        pi_d = (self.ns_dkl[d, :, :] + self.gamma) / \
               (self.ns_dk[d] + self.numSentiments * self.gamma)[:, np.newaxis]  # dim (K x L)
        return ((theta_d[:, np.newaxis] * pi_d).sum(axis=0))

    def classify_senti(self):
        doc_sent_inference = []
        for i in range(self.numDocs):
            if i in self.pos_neg_sentence_indices:
                doc_sent_inference.append(np.argmax(self.getDocSentiment(i)))
        infer_arr = np.array(doc_sent_inference)
        answer = np.array(self.pos_neg_sentiment_label)
        return np.mean(infer_arr == answer)

    def run(self, reviews, maxIters=10):
        #self._initialize_(reviews)

        for iteration in range(maxIters):
            self.updateTopicVectors()
            if (iteration + 1) % 2 == 0:
                print("Starting iteration %d of %d" % (iteration + 1, maxIters))
                print(self.classify_senti())
            for d in range(self.numDocs):
                for m in range(self.numSentence[d]):
                    t = self.topics[(d, m)]
                    s = self.sentiments[(d, m)]
                    self.ns_d[d] -= 1
                    self.ns_dkl[d, t, s] -= 1
                    self.ns_dk[d, t] -= 1
                    for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):
                        self.n_wkl[w, t, s] -= 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
                        self.n_kl[t, s] -= 1  # topic k, senti l로 할당된 단어 수

                    probabilities_ts = self.conditionalDistribution(d, m)
                    ind = sampleFromCategorical(probabilities_ts.flatten())
                    t, s = np.unravel_index(ind, probabilities_ts.shape)
                    self.topics[(d, m)] = t
                    # sentiment를 반은 supervise, 반은 sampling
                    # b = np.random.binomial(1, 0.5)
                    # s = (1 - b) * s + b * self.docSentiment[d]
                    self.sentiments[(d, m)] = s
                    self.ns_d[d] += 1
                    self.ns_dkl[d, t, s] += 1
                    self.ns_dk[d, t] += 1
                    for i, w in enumerate(word_indices(self.doc_sent_word_dict[d], m)):
                        self.n_wkl[w, t, s] += 1  # w번째 단어가 topic은 t, sentiment s로 할당된 개수
                        self.n_kl[t, s] += 1  # topic k, senti l로 할당된 단어 수


# run
work_path = "/media/hs-ubuntu/data/dataset/MasterThesis/"
data = pd.read_csv(work_path + "elec_df_preprocessed.csv",nrows =1000)
samsung = data[data['brand'] == 'Samsung']
samsung['preprocessed'] = samsung.preprocessed.apply(lambda row: literal_eval(row))
tagged_text_list = list(samsung['reviewSentence_tagged'])
sampler = STMD_Gibbs_Sampler(numTopics=10, alpha=0.01, beta=0.01, gamma=1, numSentiments=2)
sampler._initialize_(tagged_text_list)
sampler.run(tagged_text_list)