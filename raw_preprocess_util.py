%matplotlib inline
import gzip
import pandas as pd
from pandas import DataFrame
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from ast import literal_eval
from nltk.stem.porter import PorterStemmer


def parse(path):
    """
    :param path: data path
    :return: read data line by line
    """
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def load_data(path, year):
    """
    :param path: data path
    :param year: int, 연도
    :return: year 이상 data 추출
    """
    data_list = []
    for e in parse(path):
        e['overall']=float(e['overall'])
        e['unixReviewTime'] = datetime.fromtimestamp(int(e['unixReviewTime'])).strftime('%Y-%m-%d')
        if int(e['unixReviewTime'].split('-')[0]) >= year:
            row = list([e['unixReviewTime'],e['asin'],e['reviewerID'],e['overall'],e['helpful'],e['reviewText']])
            data_list.append(row)
        else:
            continue
    return data_list

def load_meta(path):
    """
    :param path: data path
    :return: asin, title, brand from meta data
    """
    meta_data = []
    for e in parse(path):
        row = list([e['asin'], e.get('title'), e.get('brand')])
        meta_data.append(row)
    return meta_data

def join_meta_data(review, meta):
    review_df = DataFrame(review, columns = ["reviewTime","asin","reviewerID","overall","helpful","reviewText"])
    meta_df = DataFrame(meta, columns = ["asin","title","brand"])
    join_data_df = pd.merge(review_df, meta_df, on = "asin", how="left")
    join_data_df.dropna(subset=['brand', 'reviewText'], how='any', inplace=True)
    join_data_df = join_data_df[join_data_df.brand != "Unknown"]
    return join_data_df

def extract_sentence(df):
    """
    :param df: meta + raw review 를 join 한 dataframe
    :return: review sentence, length of sentence, helpful score
    """
    df.dropna(subset=['brand', 'reviewText'], how='any', inplace=True)
    df = df[df.brand != "Unknown"]
    sentences = df.apply(lambda row: sent_tokenize(row['reviewText']), axis=1)
    df['reviewSentence'] = sentences
    sent_length = df.apply(lambda row: len(row['reviewSentence']), axis = 1)
    df['sent_length'] = sent_length
    df['helpful'] = df.helpful.apply(lambda x:literal_eval(x)[0])
    return df


def top_brands(data, topn=50):
    """
    get top brands dataframe (review count 기준)
    :param data: total dataframe
    :param topn: num of top brands
    :return: top brands data, top brands list
    """
    grouped = data.groupby('brand')
    s = grouped.count()
    brand_name = s.sort_values('asin', ascending=False).index[:topn]
    return data[data['brand'].isin(brand_name)], brand_name

def sort_helpful_score(data, brand, score = 5000):
    """
    해당 브랜드의 리뷰를 helpful score로 정렬했을때,
    5000개가 넘으면 상위 5000개, 아니면 전부 가져옴
    :param data: total data
    :param brand: 특정 브랜드 e.g. Samsung, Apple
    :return: 특정 브랜드의 상위 brand sorted by helpful score
    """
    brand_df = data[data.brand == brand]
    result = brand_df.sort_values('helpful', ascending=False)
    if result.shape[0] >= score:
        return result[:score]
    else:
        return result

def sample_data(data, brands):
    """
    sort_helpful_score 함수를 기준으로 정렬한 데이터를 합치는 함수
    :param data: total data
    :param brands: top brands list (from top_brands 함수)
    :return: 해당 category의 최종 sampling 데이터
    """
    top_df_list = []
    for brand in brands:
        top_df_list.append(sort_helpful_score(data, brand))
    result = pd.concat(top_df_list, axis=0)
    result.reset_index(drop=True, inplace=True)
    return result

def del_element_by_indice(lst, indice):
    """
    해당 indice에 들어가 있는 index의 원소를 제거
    """
    new_lst = []
    for index, value in enumerate(lst):
        if index-1 in indice:
            continue
        else:
            new_lst.append(value)
    return new_lst

def sentence_postag(reviewSentence):
    """
    형태소 분석 by sentence
    tokenize : nltk.word_tokenize + '.','/' 으로 분할
    """
    tokenize = [nltk.word_tokenize(sent) for sent in reviewSentence]
    tokenize2 = []
    for sent in tokenize:
        sent_token = []
        for word in sent:
            if bool(re_split.search(word)): # /, . 이 1개 이상 있으면 split
                token = re_split.split(word)
                sent_token.extend(token)
            else:
                sent_token.append(word)
        sent_token = [word for word in sent_token if len(word)>0] #길이가 0인 문자열 제거
        tokenize2.append(sent_token)
    tagged = nltk.pos_tag_sents(tokenize2)
    return tagged


def preprocessing(tagged_by_Sentence):
    """
    1. 특수문자 제거, 소문자
    2. not, "n't" -> not_stemming(다음단어)
    3. 특수문자 제거, 숫자 제거
    4. stopword 제거
    5. stemming
    """
    st = PorterStemmer()
    new_sent = []
    not_indice = []

    for sent in tagged_by_Sentence:
        text = [(tup[0].lower(), tup[1]) for tup in sent if not bool(reg_special.match(tup[0]))]  # 1. 특수문자 제거, 소문자

        # 2. not, n't 랑 다음단어 합치기
        # not, n't 가 나오면 다음 단어랑 합치고, 그 다음 단어의 index를 저장해놨다가 del_element_by_indice 함수에서 제거
        new_text = []
        for index, tup in enumerate(text):
            if tup[0] == "n't" or tup[0] == "not":
                if index + 1 < len(text):
                    if not bool(reg_special.match(text[index + 1][0])) or text[index + 1][1] != 'CD':
                        new_text.append("not_" + st.stem(text[index + 1][0]))
                        not_indice.append(index)
                else:
                    new_text.append("not")
            else:
                if not bool(reg_num.match(tup[0])) or tup[1] != 'CD': # 3. 특수문자, 숫자 제거
                    new_text.append(tup[0])
        new_text = del_element_by_indice(new_text, not_indice)

        new_words = [st.stem(word) for word in new_text if word not in stopwords]  # 4,5 stopword 제거, stemming
        new_sent.append(new_words)
    return new_sent