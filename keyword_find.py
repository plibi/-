import re
import numpy as np
import pandas as pd
# import warnings
# warnings.filterwarnings(action='ignore')

def LDA1_pre(LDA1):
    p = re.compile('[가-힣]+')
    LDA1_preprocessing = []
    for topic in LDA1:
        topic_spl = topic[1].split(' + ')
        tmp_list = []
        for i in topic_spl:
            tmp_list.append((p.findall(i)[0], float(i[:5])))
        LDA1_preprocessing.append(tmp_list)
    return LDA1_preprocessing

# 실행속도 0.0136
def normal_score(data):
    score = []
    tmp = pd.DataFrame(data[0], columns = ['word', 'score'])
    for i in data[1:]:
        tmp = pd.concat([tmp, pd.DataFrame(i, columns = ['word', 'score'])], ignore_index=True)
    tmp['score_n'] = (tmp['score'] / tmp['score'].sum()).round(6)
    for j in range(len(tmp)):
        score.append((tmp['word'][j], tmp['score_n'][j]))  
    return score

# 실행속도 0.001107
def getscore(model_result):
    # Make data as numpy array
    # shape = (10, 10, 2) = (n_topic, n_keyword, (keyword, score))
    array_data = np.array(model_result)
    total_score_sum = array_data[:, :, 1].astype(float).sum()
    # Make score list => (keyword, ratio_score)
    score_list = []
    for topic in array_data:
        for keyword, score in topic:
            ratio_score = (float(score) / total_score_sum).round(6)
            score_list.append((keyword, ratio_score))
    # Return score list
    return score_list


def normal_keybert(data):
    score = []
    tmp = pd.DataFrame(data, columns = ['word', 'score'])
    tmp['score_n'] = (tmp['score']/tmp['score'].sum()).round(6)
    for i in range(len(tmp)):
        score.append((tmp['word'][i], tmp['score_n'][i]))   
    return score


def make_dict(LDA1, LSA, keybert):
    tmp_dict = dict()
    for i, j, k in zip(LDA1, LSA, keybert):
        tmp_dict[i[0]] = i[1] if i[0] not in tmp_dict else tmp_dict[i[0]]+i[1]
        tmp_dict[j[0]] = j[1] if j[0] not in tmp_dict else tmp_dict[j[0]]+j[1]    
        tmp_dict[k[0]] = k[1] if k[0] not in tmp_dict else tmp_dict[k[0]]+k[1]
    tmp = sorted(tmp_dict.items(), key=lambda x:x[1], reverse=True)
    return tmp


def keyword_extract(LDA1, LSA, keybert, key_n):
    LDA1_tmp = LDA1_pre(LDA1)
    LDA1_normal = getscore(LDA1_tmp)
    LSA_normal = getscore(LSA)
    keybert_normal = normal_keybert(keybert.most_common(key_n))
    normal_keyword = make_dict(LDA1_normal, LSA_normal, keybert_normal)          
         
    df_LDA1 = pd.DataFrame([sorted(LDA1_normal, key=lambda x:x[1], reverse=True)]).transpose().rename(columns = {0:'LDA1'})
    df_LSA = pd.DataFrame([sorted(LSA_normal, key=lambda x:x[1], reverse=True)]).transpose().rename(columns = {0:'LSA'})
    df_keybert = pd.DataFrame([sorted(keybert_normal, key=lambda x:x[1], reverse=True)]).transpose().rename(columns = {0:'keybert'})
    df_concat= pd.concat([df_LDA1, df_LSA, df_keybert], axis=1)
    
    return df_concat, normal_keyword

# Get ensemble result
# getensemble
def Ensemble_LDA(dic_tp):
    tmp_dic = dict()
    for num in dic_tp:
        for topic in dic_tp[num]:
            for i in topic:
                tmp_dic[i[0]] = i[1] if i[0] not in tmp_dic else tmp_dic[i[0]]+i[1]
    tmp = sorted(tmp_dic.items(), key=lambda x:x[1], reverse=True)
    return tmp


def similar_word(CTM, keyword, n_word, sim_word):
    dic_ctm = dict(CTM)
    keyword_n = []
    for i in keyword[:n_word]:
        keyword_n.append(i[0])
    word_dic = dict()
    for j in keyword_n:
        for k in dic_ctm.values():
            if j in k[:sim_word]:
                word_dic[j] = k[:sim_word] if j not in word_dic else word_dic[j]+k[:sim_word]
                word_dic[j] = list(set(word_dic[j]))

                # Remove keyword in similar word list
                if j in word_dic[j]:
                    word_dic[j].remove(j)
    return word_dic


# def keyword_ratio(ensemble, percent):
#     df_tmp = pd.DataFrame(ensemble, columns=['word', 'score'])
#     df_tmp['all_ratio'] = df_tmp.score / df_tmp.score.sum() * 100
#     df_result = df_tmp[df_tmp['all_ratio']>=percent].copy()
#     df_result['ratio'] = df_result.all_ratio / df_result.all_ratio.sum() * 100
#     return df_result


def keyword_ratio(ensemble, n_keyword):
    df_tmp = pd.DataFrame(ensemble, columns=['word', 'score'])
    df_tmp['all_ratio'] = df_tmp.score / df_tmp.score.sum() * 100
    df_result = df_tmp[:n_keyword].copy()
    df_result['ratio'] = df_result.all_ratio / df_result.all_ratio.sum() * 100
    return df_result


def getratio(ensemble_result):
    df = pd.DataFrame(ensemble_result, columns=['word', 'score'])
    df['all_ratio'] = df.score / df.score.sum() * 100
    # df['ratio'] = df.all_ratio / df.all_ratio.sum() * 100
    return df

# warnings.filterwarnings(action='default')