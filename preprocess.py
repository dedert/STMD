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
    documents = [] #doc2vec에 들어갈 list, 문장단위로 긍정/부정/중립 tag
    sentence_list = [] #문장단위 list
    sentence_senti_label = {} #각 문장의 긍,부정 label
    pos_neg_sentence_indices = [] #긍, 부정 문장의 index, 중립은 None -> 나중에 긍,부정리뷰 안에 있는 문장만 추출하기 위해
    sentiment_label = [] #각 문장의 긍, 부정 index
    sentence_position = {} #key : tup(d,m), value : sentence_index
    numSentence = {} #key : doc_index, value : num of sentences in doc_index
    index = 0
    for doc_index, row in data.iterrows():
        sentence_list.extend(row['preprocessed'])
        numSentence[doc_index] = len(row[col_name])
        if row['overall']>=4:
            for sent_index, sentence in enumerate(row[col_name]):
                document = TaggedDocument(words=sentence, tags=['positive'])
                documents.append(document)
                sentence_senti_label[index]='positive'
                sentence_position[(doc_index, sent_index)] = index
                sentiment_label.append(1)
                pos_neg_sentence_indices.append(index)
                index += 1
        elif row['overall'] == 3:
            for sent_index, sentence in enumerate(row[col_name]):
                document = TaggedDocument(words=sentence, tags=['neutral'])
                documents.append(document)
                sentence_senti_label[index]='neutral'
                sentence_position[(doc_index, sent_index)] = index
                pos_neg_sentence_indices.append(None)
                index += 1
        else:
            for sent_index, sentence in enumerate(row[col_name]):
                document = TaggedDocument(words=sentence, tags=['negative'])
                documents.append(document)
                sentence_senti_label[index]='negative'
                sentence_position[(doc_index, sent_index)] = index
                sentiment_label.append(0)
                pos_neg_sentence_indices.append(index)
                index += 1
    return documents, sentence_list, sentence_senti_label, sentence_position, numSentence

def bigram_and_sentence(sentence_list, numSentence, threshold = 10):
    """
    sentence 만 들어있는 list(flatten)를 다시 문서, 문장모양의 list로 변환
    :param sentence_list: 문장 list
    :param numSentence: 각 문서의 문장 길이
    :param threshold: bigram의 threshold
    :return:
    """
    bigram = Phrases(sentences=sentence_list, threshold=threshold)
    sentence_list_again = []
    numDocs = len(numSentence.keys())
    for i in range(numDocs):
        num_sentence = numSentence[i]
        document = []
        for num_s in range(num_sentence):
            document.append(bigram[sentence_list[i+num_s]])
        sentence_list_again.append(document)
    return bigram, sentence_list_again
