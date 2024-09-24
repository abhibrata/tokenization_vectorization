import streamlit as st
import requests
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PIL import Image

image_path = "aciesglobal.png"  
image = Image.open(image_path)
st.image(image, caption="Your Static Image", use_column_width=True)


def word_tokenization(sentence):
    return sentence.split()


def sentence_tokenization(sentence):
    return sentence.split('.')

def regex_tokenization(sentence, pattern=r'\w+'):
    return re.findall(pattern, sentence)



def embed_sentence(gemini_embedder, sentence):
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embedding = embedding_model.embed_query(sentence)
        return embedding
    except Exception as e:
        st.error(f"Failed to embed the sentence: {str(e)}")
        return []


st.title("Beamed from Bengaluru episode 2: Tokenization and Embedding ")


GOOGLE_API_KEY = st.text_input("Enter your Gemini API key:", type="password")


if GOOGLE_API_KEY:
    
    sentence = st.text_input("Enter a sentence for tokenization and embedding:")

    
    tokenization_strategy = st.radio(
        "Select a tokenization strategy:",
        ["Word Tokenization", "Sentence Tokenization", "Regular Expression Tokenization"]
    )

    
    if st.button("Tokenize"):
        if sentence:
            if tokenization_strategy == "Word Tokenization":
                tokens = word_tokenization(sentence)
            elif tokenization_strategy == "Sentence Tokenization":
                tokens = sentence_tokenization(sentence)
            elif tokenization_strategy == "Regular Expression Tokenization":
                # Optionally allow the user to input a regex pattern, otherwise use the default
                pattern = st.text_input("Enter regex pattern (default: \\w+):", r'\w+')
                tokens = regex_tokenization(sentence, pattern)

            # Display tokenized result
            st.write(f"Tokens ({tokenization_strategy}):", tokens)
        else:
            st.warning("Please enter a sentence to tokenize.")


    gemini_embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001",output_dimensionality =3,task_type="retrieval_query")

   
    if st.button("Embed"):
        if sentence:
            # Perform embedding using the Gemini API
            embedding = embed_sentence(gemini_embedder, sentence)
            if embedding:
                st.write(f"Vector Embedding:", embedding)
        else:
            st.warning("Please enter a sentence to embed.")
else:
    st.warning("Please enter your Gemini API key to proceed.")
