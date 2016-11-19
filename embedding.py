import pandas as pd
from gensim.models import Doc2Vec
import random
import multiprocessing
from tqdm import tqdm

from preprocess import *

data_path = "/media/hs-ubuntu/data/dataset/Amazon/"
work_path = "/media/hs-ubuntu/data/dataset/MasterThesis/"
data = pd.read_csv(work_path + "elec_df_preprocessed.csv")


brand = 'Samsung'
brand_df = data[data['brand'] == brand]
brand_df['reviewSentence'] = brand_df.reviewSentence.apply(lambda row: literal_eval(row))
brand_df['reviewSentence_tagged'] = brand_df.reviewSentence_tagged.apply(lambda row: literal_eval(row))
brand_df['preprocessed'] = brand_df.preprocessed.apply(lambda row: literal_eval(row))
brand_df.reset_index(drop=True, inplace=True)


documents, sentence_list, sentence_senti_label, pos_neg_sentence_indices, pos_neg_sentiment_label, numSentence = prepare(brand_df)
bigram, sentence_list_again,total_token = bigram_and_sentence(sentence_list, numSentence, threshold = 15)

w = 5
s = 100
passes = 10
model = Doc2Vec(dm=1,
                dm_mean=1,
                min_count=5, sample=1e-5,
                window=w, size=s,
                workers=multiprocessing.cpu_count(),
                alpha=0.025, min_alpha=0.025)
model.build_vocab(bigram_documents)

for epoch in tqdm(range(passes)):
    random.shuffle(bigram_documents)
    model.train(bigram_documents)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay