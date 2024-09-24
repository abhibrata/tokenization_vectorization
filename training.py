import streamlit as st
import requests
import re
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PIL import Image

# Load and display the static image at the top of the page
image_path = "aciesglobal.png"  # Path to the static image
image = Image.open(image_path)
st.image(image, caption="Your Static Image", use_column_width=True)

# Function for word tokenization (Manual implementation without external libraries)
def word_tokenization(sentence):
    return sentence.split()

# Function for sentence tokenization (Manual implementation without external libraries)
def sentence_tokenization(sentence):
    return sentence.split('.')

# Function for regular expression tokenization
def regex_tokenization(sentence, pattern=r'\w+'):
    return re.findall(pattern, sentence)


# Function to embed the sentence using Gemini Embeddings via LangChain
def embed_sentence(gemini_embedder, sentence):
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embedding = embedding_model.embed_query(sentence)
        return embedding
    except Exception as e:
        st.error(f"Failed to embed the sentence: {str(e)}")
        return []

# Streamlit app UI and logic
st.title("Beamed from Bengaluru episode 2: Tokenization and Embedding ")

# Input Gemini API Key
GOOGLE_API_KEY = st.text_input("Enter your Gemini API key:", type="password")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# Ensure API key is provided
if GOOGLE_API_KEY:
    # User input for sentence to tokenize and embed
    sentence = st.text_input("Enter a sentence for tokenization and embedding:")

    # Tokenization strategies as radio buttons
    tokenization_strategy = st.radio(
        "Select a tokenization strategy:",
        ["Word Tokenization", "Sentence Tokenization", "Regular Expression Tokenization"]
    )

    # Tokenize the sentence based on the selected strategy
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

 # Initialize the Gemini Embedder with the API key
    gemini_embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001",output_dimensionality =3,task_type="retrieval_query")

    # Embed the sentence using the available vector embedding models in Gemini API
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
