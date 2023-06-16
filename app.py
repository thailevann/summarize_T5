import streamlit as st
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer
import transformers
import string
import re
import urllib.request
import torch
st.set_page_config(page_title="T5 Summarization", page_icon=":memo:", layout="wide")
st.title("T5 Summarization")
st.write("Enter your text below and click on 'Summarize' to generate a summary")
text_input = st.text_area("Text", height=200)
@st.cache_data
#@st.cache(allow_output_mutation = True)
#@st.experimental_singleton
def load_model_T5(url):
    tokenizer = AutoTokenizer.from_pretrained("minhtoan/t5-small-wikilingua_vietnamese")
    model = T5ForConditionalGeneration.from_pretrained(url)
    return tokenizer, model

def load_model_viT5(url):
    tokenizer = T5Tokenizer.from_pretrained('vietai/viT5-base')
    model.save_pretrained(url)
    model = T5ForConditionalGeneration.from_pretrained(url)
    return tokenizer, model


def generate_summary(text, tokenizer, model):
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    summary_ids = model.generate(tokenized_text, do_sample=True, max_length=1000, top_k=0,temperature=0.7)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_summary_viT5(text, tokenizer, model):
    # Tokenize input text
    inputs = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Generate summary
    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128, num_beams=2)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return summary


def preprocess(text):
  text = text.lower()
  url_pattern = re.compile(r'https?://\S+|www\.\S+')
  html_pattern = re.compile('<.*?>')
  text =  html_pattern.sub(r'', text)
  text = url_pattern.sub(r'', text)
  PUNCT_TO_REMOVE = string.punctuation
  text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
  emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
  text =  emoji_pattern.sub(r'', text)
  return text.strip()

if st.button("Summarize"):
    #tokenizer_viT5, model_viT5 = load_model_viT5('/content/drive/MyDrive/52000820 - Deep learning/Cuoi Ki - DL Application/deploy/model_deploy_viT5')
    #summary_viT5 = generate_summary_viT5("Summarize: " + preprocess(text_input), model_viT5, tokenizer_viT5)
    tokenizer_T5_9, model_T5_9 = load_model_T5('/content/drive/MyDrive/Deep_learning/Cuoi_Ki_DL/deploy/model_deploy_T5_9epoch')
    tokenizer_T5_1, model_T5_1 = load_model_T5('/content/drive/MyDrive/Deep_learning/Cuoi_Ki_DL/deploy/model_deploy_T5_1epoch')
    #tokenizer_viT5, model_viT5 = load_model_viT5('/content/drive/MyDrive/~./model_deploy_viT5')
    if text_input.strip() != "":
        with st.spinner("Summarizing your text..."):
            summary_T5_9 = generate_summary("Summarize: " + preprocess(text_input), tokenizer_T5_9, model_T5_9)
            #summary_viT5 = generate_summary_viT5("Summarize: " + preprocess(text_input), model_viT5, tokenizer_viT5)
            #summary_viT5 =  generate_summary("Summarize: " + preprocess(text_input), tokenizer_viT5, model_viT5)
            summary_T5_1 = generate_summary("Summarize: " + preprocess(text_input), tokenizer_T5_1, model_T5_1)
        st.write("T5 with 9 epoch")
        st.success(summary_T5_9)
        st.write("T5 with 1 epoch")
        st.success(summary_T5_1)
    else:
        st.warning("Please enter some text to summarize")