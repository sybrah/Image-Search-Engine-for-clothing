import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Optional
from dotenv import load_dotenv
from io import BytesIO
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, BertTokenizer, BertModel  
from sentence_transformers import SentenceTransformer
from PIL import Image
import io


blip_tokenizer = AutoTokenizer.from_pretrained("salesforce/blip-image-captioning-base")
bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")

st.set_page_config(page_title="Find Your Look", page_icon="👗", layout="wide")

st.title("👗 Find Your Look: Image Search Engine")
st.markdown("Discover your perfect fit with Find Your Look! Explore a wide variety of clothing from different Retail-Shops. Build outfits and discover new trends all in one place.")
st.divider()
load_dotenv()

# Define environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
STREAMLIT_API_KEY = os.getenv("STREAMLIT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DATA_PATH = r"./embeddings_database" 
COLLECTION_NAME = "SearchEngine"


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

def get_bert_embedding(text):
    """Generates an embedding for the given text using BERT."""
    encoded_text = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad(): 
        outputs = bert_model(**encoded_text)
    embeddings = outputs.last_hidden_state[:, 0, :] 
    embeddings_list = embeddings.squeeze(0).tolist() 
    return embeddings_list

df = pd.read_parquet('Clothes.parquet')

def search_images(query,n):

    client = chromadb.PersistentClient(CHROMA_DATA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="google-bert/bert-base-uncased",
        model_kwargs={"device": "cpu"},
    )


    langchain_chroma = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=huggingface_embeddings,
    )


    st.write(f"Searching for an image of '{query}'...")
    query_embedding = get_bert_embedding(query)
    
    with st.spinner('Finding your perfect look...'):
        results = collection.query(query_embeddings=query_embedding, n_results=n)
        original_captions = [result['original_caption'] for result in results['metadatas'][0]]
        images = []
        for result in results:
            for cap in original_captions:
                i = df[df['text'] == cap].index[0]
                image_dict = df['image'][i]
                image_bytes = image_dict["bytes"]
                image_array = Image.open(io.BytesIO(image_bytes))
                images.append(image_array)

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for idx, (col, result) in enumerate(zip(cols, results['metadatas'][0])):
        with col:
            st.image(images[idx], use_column_width=True)
            st.write( result['original_caption'])

    return results


def main():
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None  

    query = st.text_input("Describe the image of the item you're looking for:", key="query", placeholder="e.g., Red summer dress with floral patterns")

    if query:
        results = search_images(query,4)

        st.header("Feedback")
        satisfaction = st.selectbox("Are you satisfied with the results?", ("YES", "NO"))
        if satisfaction == "YES":
            st.success("Great!")
        elif satisfaction == "NO":
            choice = st.selectbox("Specify the brand", (
                "Louis Vuitton", "Gucci", "Balenciaga", "Off-White", "Maison Martin Margiela", "Prada", "Belstaff", 
                "Calvin Klein", "Alexander McQueen", "Rolex", "Jacquemus", "Saint Laurent", "Chanel", "Tom Ford", 
                "Valentino", "Moncler", "Berluti", "Dior", "Gianni Versace", "Loro Piana", "Comme Des Garcons", 
                "Casablanca", "Dries Van Noten", "Givenchy", "Bottega Veneta", "Christian Louboutin", "Balmain", 
                "Ralph Lauren", "Brunello Cucinelli", "Acne Studios", "Sandro", "Yeezy x Adidas", "Giorgio Armani", 
                "Nike", "Mulberry", "Etro", "Stone Island", "Cartier", "Loewe", "Fendi", "Celine", "Ami", "Hermès", 
                "Craig Green", "Aspinal Of London", "Canada Goose", "Goyard", "Palm Angels", "Thom Browne", 
                "Maharishi", "Amiri", "Supreme", "Kiko Kostadinov", "Zilli"
            ))
            
            if choice:
                query_with_brand = f"{choice} {query} "
                st.write(f"Re-running search with brand: {choice}")
                search_images(query_with_brand,1)

if __name__ == "__main__":
    main()