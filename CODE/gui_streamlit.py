import numpy as np
import pandas as pd
import streamlit as st
from recommend_sys import Recommendation_System
from project_transformer import Data_Wrangling
import time
# How to load image
import re
from utilities import get_dict_word, markdown_insert_images, view_info_prd

# ----------------- INITIALIZATION -----------------
emoji_dict = get_dict_word(file_path='DATA/files/emojicon.txt')  # Emoji
teen_dict = get_dict_word(file_path='DATA/files/teencode.txt')  # Teen
eng_dict = get_dict_word(file_path='DATA/files/english-vnmese.txt')  # eng

with open('DATA/files/wrong-word.txt', 'r', encoding="utf8") as file:  # WRONG WORD
    wrong_lst = file.read().split('\n')

with open('DATA/files/vietnamese-stopwords.txt', 'r', encoding="utf8") as file:  # STOP WORD
    stop_lst = file.read().split('\n')

# GET CLEANSER
cleanser = Data_Wrangling(emoji_dict=emoji_dict,
                          teen_dict=teen_dict,
                          wrong_lst=wrong_lst,
                          eng_vn_dict=eng_dict,
                          stop_words=stop_lst)
# Model
gensim_model = 'DATA/Gensim/tfidf_model'
gensim_dict = 'DATA/Gensim/corpus_dictionary'
gensim_matrix = "DATA/Gensim/similarity_matrix.index"
cos_model = 'DATA/tfidf_vectorizer.pkl'
cos_matrix = 'DATA/tfidf_matrix.npz'
surprise_path = "DATA/Surprise/col_fitering_surprise.pkl"

# Use Columns
use_cols = ['item_id', 'product_id', 'sub_category', 'price', 'rating',
            'clean_desc', 'product_name', 'clean_prd_name', 'link', 'image']

# MarkDown File
md_path = 'DATA/report_md/recommendation_sysyten_report.md'
with open(md_path, 'r', encoding='utf-8') as file:
    markdown_text = file.read()

readme_img = markdown_insert_images(markdown_text)


# ----------------- ADD CACHE -----------------
# Cache the model Recommendation_System
@st.cache_resource()
def load_model():
    return Recommendation_System(cleanser=cleanser,
                                 prd_dataset=clean_prd_df,
                                 gensim_model=gensim_model,
                                 gensim_dict=gensim_dict,
                                 gensim_matrix=gensim_matrix,
                                 cos_model=cos_model,
                                 cos_matrix=cos_matrix,
                                 surprise_path=surprise_path)


@st.cache_resource()
def get_prd_df(file_path, use_cols):
    df = pd.read_csv(filepath_or_buffer=file_path,
                     usecols=use_cols)
    return df


def stream_data(df):
    # Stream the dataframe
    _Overview = """
    This is overview about Product Data. I will show a sample of 10 products
    """
    for word in _Overview.split(" "):
        yield word + " "
        time.sleep(0.02)

    # get dataframe
    sample_ = df.sample(10).drop(columns=['clean_desc', 'clean_prd_name'])
    # sample['image'] = sample['image'].apply(lambda x: f"[Click here]({x})")
    # sample['link'] = sample['link'].apply(lambda x: f"[Click here]({x})")
    yield sample_


# ----------------- BUILD GUI -----------------

# Load dataframe and model from cache
clean_prd_df = get_prd_df(file_path='DATA/final_clean_details.csv', use_cols=use_cols)
# clean_de_df = get_prd_df(file_path='DATA/clean_details.csv',use_cols=['product_id','user_id','rating'])
recom_model = load_model()

menu = ['EXPLORE DATA ANALYSIS', 'COLLABORATIVE FILTERING', 'CONTENT-BASE FILTERING']
choice = st.sidebar.selectbox('MENU', menu)

if choice == 'EXPLORE DATA ANALYSIS':
    # Render the Markdown content in the Streamlit app
    with st.container():
        st.markdown(readme_img, unsafe_allow_html=True)

# --------------- COLLABORATIVE FILTERING ---------------
elif choice == 'COLLABORATIVE FILTERING':
    st.title("Recommend Product by userID")
    if st.button("View Product data"):
        st.write_stream(stream_data(df=clean_prd_df))

    # ls_userID = clean_de_df['user_id'].unique().tolist()
    st.write("### :green[Please type user ID you want 👇]")
    st.write(":white[Please type number only, If you have UserID, you can type it. On the other hand, please type 0]")
    user_id_select = st.text_input(label=" ")
    # user_id_select_test = st.selectbox(label="",
    #                                    options=ls_userID,
    #                                    index=None,
    #                                    placeholder='Please choose the UserID or leave it None')

    if re.search(pattern=r'\d+', string=user_id_select):
        # Product recommendation:
        prd_by_user = recom_model.col_fil_surprise(user_id=int(user_id_select))
        st.dataframe(prd_by_user)

        # View Product
        sel_prd_ids = prd_by_user['product_id'].tolist()
        sample = clean_prd_df[clean_prd_df['product_id'].isin(sel_prd_ids)].to_dict(orient='record')
        if st.button("View Info of each product"):
            view_info_prd(list_prd=sample)
    else:
        st.write("Please type userID with correct format")

# --------- CONTENT BASE FILTERING ---------------
elif choice == 'CONTENT-BASE FILTERING':
    st.title("Recommend Product by Content")
    if st.button("Overview Product data"):
        st.write_stream(stream_data(df=clean_prd_df))

    st.write("### :blue[Please type product you want 👇]")
    content = st.text_input(label=" ")
    if re.search(pattern=r'[a-zA-Z]+', string=content):
        st.write(f"Content User type: {content}")
        # Product recommendation:
        gensim_df = recom_model.content_base_gensim(finding_text=content)
        cosine_df = recom_model.content_base_cosine(finding_text=content)
        combined_df = recom_model.content_base_rcmd(cosim_df=cosine_df, gensim_df=gensim_df)
        st.dataframe(combined_df)

        # View Product
        sel_ids = combined_df['product_id'].tolist()
        sample_cb = clean_prd_df[clean_prd_df['product_id'].isin(sel_ids)].to_dict(orient='record')
        if st.button("View Info of each product"):
            view_info_prd(list_prd=sample_cb)
    else:
        st.write("Please type content more clearly")
