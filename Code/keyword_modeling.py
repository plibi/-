from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

import pandas as pd
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel

from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.gensim_models
import tomotopy as tp

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer

from konlpy.tag import Mecab
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation, bert_embeddings_from_list


############################# Coherence & Peplexity ##############################
def coherence_function(df, max_topic, top_n):
    dic_below = int(len(df)*0.0001)
    dictionary = corpora.Dictionary(df)
    dictionary.filter_extremes(no_below = dic_below)
    corpus = [dictionary.doc2bow(text) for text in df]
    coherence_list = []
    for i in range(2, max_topic):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = i)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts = df, dictionary = dictionary, topn=top_n)
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_list.append(coherence_lda)
    return coherence_list


def perplexity_function(df, max_topic):
    dic_below = int(len(df)*0.0001)
    dictionary = corpora.Dictionary(df)
    dictionary.filter_extremes(no_below = dic_below)
    corpus = [dictionary.doc2bow(text) for text in df]
    perplexity_list = []
    for i in range(2, max_topic):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = i, id2word = dictionary)
        perplexity_list.append(ldamodel.log_perplexity(corpus))
    return perplexity_list


###################################### LSA ######################################
def get_topics(components, feature_names, n=5):
    tmp = []
    for idx, topic in enumerate(components):
        tmp.append([(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])
    return tmp

def keyword_LSA(df, max_feature, n_topics, n_words, random_state = 999):
    detokenized_doc = []
    for i in range(len(df)):
        tmp = ' '.join(df[i])
        detokenized_doc.append(tmp)
        
    vectorizer = TfidfVectorizer(max_features = max_feature, max_df = 0.5, smooth_idf=True)
    X = vectorizer.fit_transform(detokenized_doc)
    
    svd_model = TruncatedSVD(n_components = n_topics, algorithm='randomized', n_iter=100, random_state = random_state)
    svd_model.fit(X)
    terms = vectorizer.get_feature_names() # 단어 집합.
    topics = get_topics(svd_model.components_, terms, n_words)
    return topics


###################################### LDA ######################################
def keyword_LDA1(df, dic_below, n_topics, n_words, passes, random_state = 999):
    dictionary = corpora.Dictionary(df)
    dictionary.filter_extremes(no_below = dic_below)
    corpus = [dictionary.doc2bow(text) for text in df]
    
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = n_topics, id2word = dictionary, passes = passes, random_state= random_state)
    topics = ldamodel.print_topics(num_words=n_words)
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
    return topics, vis


def keyword_LDA2(df, max_feature, n_topics, n_words, random_state = 999):
    detokenized_doc = []
    for i in range(len(df)):
        t = ' '.join(df[i])
        detokenized_doc.append(t)
    detokenized_doc = pd.DataFrame({'detoken':detokenized_doc})
    
    vectorizer = TfidfVectorizer(max_features= max_feature)
    X = vectorizer.fit_transform(detokenized_doc['detoken'])

    lda_model = LatentDirichletAllocation(n_components=n_topics,learning_method='online', max_iter=20, random_state = random_state)
    lda_model.fit_transform(X)
    terms = vectorizer.get_feature_names()
    topics = get_topics(lda_model.components_, terms, n_words)
    
    return topics


def LDA1_many(sentiment, df, dic_below, n_topics, n_words, passes, n_random):
    sent_dic = {0:'topics_LDA1_negative',
                1:'topics_LDA1_positive',
               100:'topics_LDA1_neutral'}
    
    globals()[f'{sent_dic[sentiment]}_many'] = dict()
    tmp_dic = dict()
    
    for random in range(0, n_random):
        topic, _ = keyword_LDA1(df, dic_below, n_topics, n_words, passes, random_state = random)
        tmp_dic[random] = topic
    
    return  tmp_dic


def LDA_tomoto(docs, n_topics, n_trains, iter, top_n=10):
    # result = {}
    min_cf = int(len(docs)*0.0001)
    models = {}
    for i in range(iter):
        # Define LDA model
        elda = tp.LDAModel(k=n_topics, min_cf=min_cf, seed=i)

        # Add document
        for doc in docs:
            elda.add_doc(doc)

        # Training
        elda.train(n_trains)

        models[i] = []
        # Save model results and scores
        for n in range(elda.k):
            models[i].append(elda.get_topic_words(n, top_n=top_n))
    # Return model results
    return models


###################################### keybert ######################################
def keyword_keybert(df, top_n):
    detokenized_doc = []
    for i in range(len(df)):
        t = ' '.join(df[i])
        detokenized_doc.append(t)
    detokenized_doc = pd.DataFrame({'detoken':detokenized_doc})    
    
    n_gram_range = (1, 1)
    count = CountVectorizer(ngram_range=n_gram_range).fit(detokenized_doc['detoken'])
    candidates = count.get_feature_names()
    
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    doc_embedding = model.encode(detokenized_doc['detoken'])
    candidate_embeddings = model.encode(candidates)
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
   
    keywords = []
    for dist in distances:
        for index in dist.argsort()[-top_n:]:
            keywords.append(candidates[index])
            
    count_list = Counter(keywords[0])
    for word in tqdm(keywords[1:]):
        count_list += Counter([word])
    return count_list


###################################### CTM ######################################
class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result


def keyword_CTM(df, max_feature, n_topic, num_epoch):
    detokenized_doc = []
    for i in range(len(df)):
        t = ' '.join(df[i])
        detokenized_doc.append(t)

    custom_tokenizer = CustomTokenizer(Mecab())
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features = max_feature)
    train_bow_embeddings = vectorizer.fit_transform(detokenized_doc)
    vocab = vectorizer.get_feature_names()
    id2token = {k: v for k, v in zip(range(0, len(vocab)), vocab)}

    train_contextualized_embeddings = bert_embeddings_from_list(detokenized_doc, "jhgan/ko-sroberta-multitask")
    qt = TopicModelDataPreparation()
    training_dataset = qt.load(train_contextualized_embeddings, train_bow_embeddings, id2token)

    ctm = CombinedTM(bow_size=len(vocab), contextual_size=768, n_components=n_topic, num_epochs = num_epoch)
    ctm.fit(training_dataset)
    # topics = ctm.get_topics(n_words)
    lda_vis_data = ctm.get_ldavis_data_format(vocab, training_dataset, n_samples=10)
    return ctm, lda_vis_data