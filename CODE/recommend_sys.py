# from CODE.project_transformer import Data_Wrangling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
import pickle
import scipy.sparse as ss

import warnings

warnings.filterwarnings("ignore")


class Recommendation_System:
    def __init__(self,
                 cleanser,
                 prd_dataset: pd.DataFrame,
                 gensim_model: str = '../DATA/Gensim/tfidf_model',
                 gensim_dict: str = '../DATA/Gensim/corpus_dictionary',
                 gensim_matrix: str = "../DATA/Gensim/similarity_matrix.index",
                 cos_model: str = '../DATA/tfidf_vectorizer.pkl',
                 cos_matrix: str = '../DATA/tfidf_matrix.npz',
                 surprise_path: str = "../DATA/Surprise/col_fitering_surprise.pkl"):
        self.cleanser = cleanser
        self.df_prd = prd_dataset
        # --------- GENSIM ----------
        # Load the TF-IDF model from the file
        self.gensim_tfidf_model = models.TfidfModel.load(gensim_model)
        # Load the dictionary from the file
        self.gensim_dictionary = corpora.Dictionary.load(gensim_dict)
        # Load the similarity matrx
        self.gensim_sim_matrix = similarities.SparseMatrixSimilarity.load(gensim_matrix)

        # --------- COSINE SIMILARITY ----------
        with open(cos_model, 'rb') as f:
            self.sim_tfidf_model = pickle.load(f)
        self.cosine_tfidf_matrix = ss.load_npz(cos_matrix)

        # --------- Load Surprise model from disk ---------
        self.loaded_model = pickle.load(open(surprise_path, 'rb'))

    # Gensim
    def content_base_gensim(self, finding_text: str):
        # Convert search words into Sparse Vectors
        clean_txt = self.cleanser.process_text(text=finding_text)
        clean_txt = self.cleanser.process_postag_thesea(clean_txt)

        clean_ls = clean_txt.split()
        kw_vector = self.gensim_dictionary.doc2bow(clean_ls)
        print("View product's vector:")
        print(kw_vector)

        # similarity calculation
        sim = self.gensim_sim_matrix[self.gensim_tfidf_model[kw_vector]]

        # print result
        list_id = []
        list_score = []
        for i in range(len(sim)):
            list_id.append(i)
            list_score.append(sim[i])

        # Create df
        df_res = pd.DataFrame({'item_id': list_id, 'score': list_score})

        # Get 5 highest score
        five_h_score = df_res.sort_values(by='score', ascending=False).head(6)
        print("Five highest scores:")
        print(five_h_score)
        print('IDS to list:')
        id_to_ls = five_h_score['item_id'].to_list()
        print(id_to_ls)

        # Find prd
        prd_find = self.df_prd[self.df_prd.index.isin(id_to_ls)]
        prd_results = prd_find[['product_id', 'product_name', 'clean_prd_name', 'sub_category', 'price', 'rating']]
        final = pd.concat([prd_results, five_h_score], axis=1).sort_values(by='score', ascending=False)
        final = final[final['score'] != 1]
        return final

    # Cosine Similarity

    def content_base_cosine(self,
                            finding_text,
                            num=5):

        # Convert search words into Sparse Vectors
        clean_txt = self.cleanser.process_text(text=finding_text)
        clean_txt = self.cleanser.process_postag_thesea(clean_txt)

        # Get Sparse Matrix from TFIDF
        txt_sparse_matrix = self.sim_tfidf_model.transform([clean_txt])

        # Calcualte the cosine score
        cosine_text_ = cosine_similarity(X=txt_sparse_matrix,
                                         Y=self.cosine_tfidf_matrix)

        top5_prd = cosine_text_.flatten().argsort()[-num - 1:-1]
        top5_score = [(cosine_text_.flatten()[i]) for i in top5_prd]

        # Extract Product
        ls_item = []
        for id, sim in zip(top5_prd, top5_score):
            sim_item = {'item_id': id, 'score': sim}
            ls_item.append(sim_item)

        print(f"User type: {finding_text}\n\n")
        for rec in ls_item:
            # print(rec[1])
            print(f"Recommnended:\tItem ID: {rec['item_id']}, {self.df_prd.loc[self.df_prd['item_id'] == rec['item_id'], 'product_name'].values[0]}, (score: {rec['score']})\n")

        # Create DataFrame
        df_res = pd.DataFrame(ls_item)
        filter_ls = df_res['item_id'].to_list()
        prd_find = self.df_prd[self.df_prd['item_id'].isin(filter_ls)]
        prd_results = pd.merge(left=prd_find,
                               right=df_res,
                               on=['item_id'],
                               how='inner')

        final = prd_results.sort_values(by='score', ascending=False)
        # final = final[final['score']!=1]
        feat = ['product_id', 'product_name', 'clean_prd_name', 'sub_category', 'price', 'rating', 'item_id', 'score']
        return final[feat]

    def content_base_rcmd(self, cosim_df, gensim_df):
        combine_df = (pd.merge(cosim_df,
                               gensim_df,
                               on=['product_id', 'product_name', 'clean_prd_name', 'sub_category', 'price', 'rating'],
                               suffixes=('_gensim', '_simi'),
                               how='outer')
                      .drop(columns=['item_id_gensim', 'item_id_simi']))
        combine_df = combine_df[combine_df['rating'] > 3]
        return combine_df

    # -------------- COLLABORATIVE FILTERING --------------
    def col_fil_surprise(self, user_id: int):
        # Recommend UserID
        df_score = self.df_prd[["product_id"]].drop_duplicates(subset='product_id').reset_index(drop=True)
        df_score['EstimateScore'] = df_score['product_id'].apply(lambda x: self.loaded_model.predict(user_id, x).est)  # est: get EstimateScore
        df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)

        # Get Top 10 and rating of prd have to greater >4
        top_10_prd = df_score.head(10)
        final_df = (self.df_prd[self.df_prd['product_id'].isin(top_10_prd['product_id'].to_list())]
                    .sort_values(by=['price', 'rating'], ascending=[True, False])
                    .drop(columns=['clean_desc', 'clean_prd_name', 'item_id']))
        return final_df


if __name__ == '__main__':
    from CODE.project_transformer import Data_Wrangling

    # ------------- INITIALIZATION -------------
    # EMOJI
    with open('../DATA/files/emojicon.txt', 'r', encoding="utf8") as file:
        emoji_lst = file.read().split('\n')
        emoji_dict = {}
        for line in emoji_lst:
            key, value = line.split('\t')
            emoji_dict[key] = str(value)

    # TEEN CODE
    with open('../DATA/files/teencode.txt', 'r', encoding="utf8") as file:
        teen_lst = file.read().split('\n')
        teen_dict = {}
        for line in teen_lst:
            key, value = line.split('\t')
            teen_dict[key] = str(value)

    # ENG VIET
    with open('../DATA/files/english-vnmese.txt', 'r', encoding="utf8") as file:
        eng_lst = file.read().split('\n')
        eng_dict = {}
        for line in eng_lst:
            key, value = line.split('\t')
            eng_dict[key] = str(value)

    # WRONG WORD
    with open('../DATA/files/wrong-word.txt', 'r', encoding="utf8") as file:
        wrong_lst = file.read().split('\n')

    # STOP WORD
    with open('../DATA/files/vietnamese-stopwords.txt', 'r', encoding="utf8") as file:
        stop_lst = file.read().split('\n')

    # GET CLEANSER
    cleanser = Data_Wrangling(emoji_dict=emoji_dict,
                              teen_dict=teen_dict,
                              wrong_lst=wrong_lst,
                              eng_vn_dict=eng_dict,
                              stop_words=stop_lst)
    use_cols = ['item_id', 'product_id', 'sub_category', 'price', 'rating', 'clean_desc', 'product_name', 'clean_prd_name','link','image']
    clean_prd_df = pd.read_csv('../DATA/final_clean_details.csv', usecols=use_cols)
    recom_sys = Recommendation_System(cleanser=cleanser, prd_dataset=clean_prd_df)

    # ------------- GENERALIZATION For CONTENT-BASE FILTERING -------------
    search = 'đồ Thể thao'
    gensim_df = recom_sys.content_base_gensim(finding_text=search)
    cosine_df = recom_sys.content_base_cosine(finding_text=search)

    print(gensim_df.shape)
    print(cosine_df.shape)
    combined_df = recom_sys.content_base_rcmd(cosim_df=cosine_df, gensim_df=gensim_df)

    print("CONTENT-BASE FILTERING:")
    print(combined_df.to_string(), '\n\n')

    # ------------- GENERALIZATION For COLLABORATE FILTERING -------------
    model_surprise_path = "../DATA/Surprise/col_fitering_surprise.pkl"
    user_id = 100
    prd_by_user = recom_sys.col_fil_surprise(user_id=100)

    print("COLLABORATE FILTERING:")
    print(prd_by_user.to_string())
