import streamlit as st
import time
import nltk
import string
from heapq import nlargest
import spacy,os
from dotenv import load_dotenv
from spacy import displacy

import requests
# Load environment variables from the .env file
load_dotenv()
# st.write(st.session_state)
HUGGING_FACE_TOKEN = os.getenv("HF_API_KEY")


API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# API_URL = "https://api-inference.huggingface.co/models/tuner007/pegasus_summarizer"
headers = {"Authorization": "Bearer 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"}

API_TOPIC_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
topic_headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}

def summarize(text):
  if text.count(". ") > 20:
      length = int(round(text.count(". ")/10, 0))
  else:
      length = 1

  nopuch =[char for char in text if char not in string.punctuation]
  nopuch = "".join(nopuch)

  processed_text = [word for word in nopuch.split() if word.lower() not in nltk.corpus.stopwords.words('english')]

  word_freq = {}
  for word in processed_text:
      if word not in word_freq:
          word_freq[word] = 1
      else:
          word_freq[word] = word_freq[word] + 1

  max_freq = max(word_freq.values())
  for word in word_freq.keys():
      word_freq[word] = (word_freq[word]/max_freq)

  sent_list = nltk.sent_tokenize(text)
  sent_score = {}
  for sent in sent_list:
      for word in nltk.word_tokenize(sent.lower()):
          if word in word_freq.keys():
              if sent not in sent_score.keys():
                  sent_score[sent] = word_freq[word]
              else:
                  sent_score[sent] = sent_score[sent] + word_freq[word]

  summary_sents = nlargest(length, sent_score, key=sent_score.get)
  summary = " ".join(summary_sents)
  return summary

@st.cache_resource
def load_spacy_depend():
    import subprocess,sys
    # print(subprocess.run([sys.executable, "-m", "spacy", "download", 'en_core_web_md'], text=True))
    nltk.download('stopwords')
    nltk.download('punkt')

# @st.cache_resource 
# def load_topic_transfomers():
#   from transformers import pipeline
#   try:
#       topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device="cuda", compute_type="float16")
#   except Exception as e:
#       topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#       print("Error: ", e)
#   return topic_classifier

# if 'topic_model' not in st.session_state:
#     with st.spinner("Loading Model....."):
#         st.session_state.topic_model =  load_topic_transfomers() 
#         st.success("Model_loaded")
#     st.session_state.model = True


# def suggest_topic(topic_classifier,text):

#     # while len(text)> 1024:
#     #     text = summarize(whole_text[:-10])

#     possible_topics = ["Gadgets", 'Business','Finance', 'Health', 'Sports',  'Politics','Government','Science','Education', 'Travel', 'Tourism', 'Finance & Economics','Market','Technology','Scientific Discovery',
#                       'Entertainment','Environment','News & Media', "Space,Universe & Cosmos", "Fashion", "Manufacturing and Constructions","Law & Crime","Motivation", "Development & Socialization",  "Archeology"]

#     result = topic_classifier(text, possible_topics)

#     return result['labels']
  
# try:
#     if st.button('Other Suggest topic'):
#         start= time.time()  
#         with st.spinner("Scanning content to suggest topics"):
#             topic_classifier = st.session_state.topic_model 
#             predicted_topic = suggest_topic(topic_classifier,whole_text)
#         clk = time.time()-start
#         if clk> 60:
#             st.write(f'Generated in {(time.time()-start)} secs')
#         else:
#             st.write(f'Generated in {(clk)/60} minutes')
            
#         st.subheader('Top 10 Topics related to the content')
#         for i in predicted_topic[:10]:
#             st.write(i)
# except Exception as e:
#     print("Error", e)


if 'nlp' not in st.session_state:
    load_spacy_depend()
    print('Creating nlp in session_state and loading it..............')
    nlp = spacy.load("en_core_web_md")
    # nlp.add_pipe('spacytextblob')
    st.session_state.nlp = nlp
    print("NLP loaded from SpaCy in the transcription video info")

st.title("Topic Suggestion")

whole_text = st.text_input("Enter the text Here: ")

if 'text' in st.session_state:
    if st.session_state.text !=whole_text:
        st.session_state.text= None
        st.session_state.text_summ= None
        st.session_state.topics= None
  

def topic_query(payload):
	topic_response = requests.post(API_TOPIC_URL, headers=topic_headers, json=payload)
	return topic_response.json()

def query(payload, api= API_URL):
	response = requests.post(api, headers=headers, json=payload)
	return response.json()

if 'topics' not in st.session_state:
    st.session_state.topics = None

    
def summarize_option(text):
    
    if len(summarize(text))>1500:
      text = summarize(text)
    try:
        print('\ntrying with Bart summarizer')
        output = query({
            "inputs":text,
        })
        if type(output) == dict and 'error' in output.keys() :
          print('\nPegasus Summarizer')
          output = query({
              "inputs": text[:10000],
          },api="https://api-inference.huggingface.co/models/tuner007/pegasus_summarizer")
          print(output,type(output))
          
          if type(output) == dict and 'error' in output.keys() :
              print("\nsummaring my function")
              return summarize(text)
    except Exception as e:
        print(e)
        print('Normal summary')
        return summarize(text)
    return output[0]["summary_text"]    


def topic_suggest_option(text):
    
    if st.session_state.text_summ is None:
        text = summarize_option(text)
        st.session_state.text_summ =  text     
     
    text = st.session_state.text_summ
    result = topic_query({
        "inputs": text,
        "parameters": {
            "candidate_labels": ['Business & Market & Finance & Economics', 'Health',  'Education & Science',
                                 'Politics & Government','Travel & Tourism','Gadgets & Technology',
                                 'Scientific Discovery & Space',"Fashion & Social Media, Law & Crime" ,
                                 'Entertainment',"Environment,Development & Socialization"
                                 ]},
        
    })
    return result["labels"][:5]

def show_timing(start):
    clk = time.time()-start
    if clk < 60:
        st.write(f'Generated in {(time.time()-start)} secs')
    else:
        st.write(f'Generated in {(clk)/60} minutes')

try:
    cols = st.columns(2)  
    with cols[0]:
        if st.button('Summarize',use_container_width=True):
            st.session_state.topics = None
            start= time.time()  
            with st.spinner('Summarizing...'):
                st.session_state.text = whole_text
                st.session_state.text_summ =  summarize_option(whole_text)
            show_timing(start)

                        
    with cols[1]:
        if st.button('suggest topics',use_container_width=True):
            
            start= time.time()  
            with st.spinner('Scaning the content to suggest the topics...'):
                st.session_state.text = whole_text
                st.session_state.topics =  topic_suggest_option(whole_text)
            show_timing(start)
except Exception as e:
    print("Error", e)   
               
if st.session_state.text_summ is not None:
    
    summ_text = st.session_state.text_summ
    tabs = st.tabs(['Description','Name Entity Recognition - (NER)'])
    with tabs[0]:
        show = st.radio('Original Text : ',  options=['hide','show',])
        if show =='show':
            st.write(st.session_state.text)
        st.subheader('Summarized Text : ')
        st.write(summ_text)
        
        print('\nSummarized Text : ')
        print(summ_text)
                
        if st.session_state.topics is not None:
            st.subheader('Topics related to the content are :  ')
            print('\nTopics related to the content are :  ')
            for topic in st.session_state.topics:
                st.write(topic)
                
                print(topic)
                
    with tabs[1]:
        tab1,tab2 = st.tabs(["Summarized Subtile Text","Whole Text"])
        nlp = st.session_state.nlp
        whole_text = st.session_state.text
        
        with tab1:
            summ_doc = nlp(st.session_state.text_summ)
            st.write("\nNER on Summarized Transcribe Text: \n")
            st.markdown(displacy.render(summ_doc, style="ent",jupyter=False), unsafe_allow_html=True)

        with tab2:
            doc = nlp(whole_text)
            st.write("NER on whole Subtile Text from video : \n")
            st.markdown(displacy.render(doc, style="ent",jupyter=False), unsafe_allow_html=True)
