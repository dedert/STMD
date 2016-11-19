import numpy as np
from gensim.models import Phrases
from gensim.models.doc2vec import TaggedDocument

def prepare(data, col_name ='preprocessed'):
    """
    전처리 완료된 문서에서(after raw_preprocess)
    기본적인 자료구조 생성
    :param data: 특정 브랜드(리뷰)의 dataframe
    :param col_name: 전처리 된 데이터가 들어있는 column name
    :return: documents, sentence_list, sentence_senti_label, sentence_position, numSentence
    """
    sentence_list = [] #문장단위 list
    sentence_senti_label = {} #각 문장의 긍,부정 label
    pos_neg_sentence_indices = [] #긍, 부정 문장의 index, 중립은 None -> 나중에 긍,부정리뷰 안에 있는 문장만 추출하기 위해
    sentiment_label = [] #각 문장의 긍, 부정 index
    pos_neg_sentiment_label = [] #sentiment_label에 해당하는 문장의 긍,부정 labe(1:긍정, 0:부정)
    numSentence = {} #key : doc_index, value : num of sentences in doc_index
    index = 0
    for doc_index, row in data.iterrows():
        sentence_list.extend(row['preprocessed'])
        numSentence[doc_index] = len(row[col_name])
        if row['overall']>=4:
            pos_neg_sentence_indices.append(doc_index)
            pos_neg_sentiment_label.append(1)
            for sent_index, sentence in enumerate(row[col_name]):
                sentence_senti_label[index]='positive'
                sentiment_label.append(1)
                index += 1
        elif row['overall'] == 3:
            pos_neg_sentence_indices.append(None)
            for sent_index, sentence in enumerate(row[col_name]):
                sentence_senti_label[index]='neutral'
                index += 1
        else:
            pos_neg_sentence_indices.append(doc_index)
            pos_neg_sentiment_label.append(0)
            for sent_index, sentence in enumerate(row[col_name]):
                sentence_senti_label[index]='negative'
                sentiment_label.append(0)
                index += 1
    return sentence_list, sentence_senti_label, pos_neg_sentence_indices, pos_neg_sentiment_label, numSentence

def bigram_and_sentence(sentence_senti_label, sentence_list, numSentence, threshold = 10):
    """
    sentence 만 들어있는 list(flatten)를 다시 문서, 문장모양의 list로 변환
    :param sentence_list: 문장 list
    :param numSentence: 각 문서의 문장 길이
    :param threshold: bigram의 threshold
    :return:
    """
    bigram = Phrases(sentences=sentence_list, threshold=threshold)
    total_token = []
    documents = []
    sentence_list_again = []
    numDocs = len(numSentence.keys())
    for i in range(numDocs):
        num_sentence = numSentence[i]
        doc_list = []
        for num_s in range(num_sentence):
            bi = bigram[sentence_list[i+num_s]]
            doc_list.append(bi)
            document = TaggedDocument(words=bi, tags=[sentence_senti_label[i]])
            documents.append(document)
            total_token.append(bi)
        sentence_list_again.append(doc_list)
    return documents, sentence_list_again, bigram, total_token


