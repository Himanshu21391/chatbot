import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma  # Updated vector store class
from transformers import BertTokenizer, BertForTokenClassification, RobertaTokenizer, RobertaForTokenClassification, pipeline
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Set the environment variable to disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define variables
docs_folder = os.path.join(os.path.dirname(__file__), 'docs')
chunk_size = 100000
chunk_overlap = 4
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
hf_api_token = "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"
openai_api_key = "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"

# Load and process documents
documents = []
for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(docs_folder, filename), encoding='utf-8')
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
docs = text_splitter.split_documents(documents)

# Create an in-memory vector store
vector_store = Chroma.from_documents(docs, embeddings)

def retrieve_similar_docs(query, top_k=5):
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = vector_store.get_embeddings()
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-top_k:]
    return [docs[i] for i in top_k_indices]

llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50, "max_new_tokens": 10000},
    huggingfacehub_api_token=hf_api_token,
)

template = """
You are a Product Recommender which finds products made by companies like STMicroelectronics and other similar companies. Additionally, you provide information of products along with availability of these products in stores along with websites or places on the internet where the products are being sold. Your role is to suggest products, their specifications, and indicate if they are Not Recommended for New Design (NRND). You should also suggest multiple products if available.

Below is a question from a user who wants to understand more about product recommendations based on the analysis of products available. Your role is to provide a factual, unbiased answer based on the provided data. If the information is not available in the data, it's okay to say you don't know. Keep your answers precise and do not answer questions not related to product recommendations.

Context: {context}
Query: {question}
Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

rag_chain = (
    {"context": retrieve_similar_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Initialize BERT and RoBERTa models
bert_model_name = "bert-base-uncased"
roberta_model_name = "roberta-base"

bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForTokenClassification.from_pretrained(bert_model_name)

# Set clean_up_tokenization_spaces explicitly to suppress future warnings
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name, clean_up_tokenization_spaces=True)
roberta_model = RobertaForTokenClassification.from_pretrained(roberta_model_name)

# Set up NLP pipelines
ner_pipeline = pipeline("ner", model=bert_model, tokenizer=bert_tokenizer)
sentiment_pipeline = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer)

# Initialize OpenAI API
openai.api_key = openai_api_key

def get_openai_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Streamlit app for handling user input
def main():
    st.title("Product Recommendation Chatbot")
    
    # User message input
    user_message = st.text_input("Enter your query:")

    if st.button("Ask"):
        if not user_message:
            st.error("Please provide a message.")
        else:
            try:
                # Retrieve context from in-memory vector store
                context_docs = retrieve_similar_docs(user_message)
                context = " ".join(doc.text for doc in context_docs)

                # Generate response using LangChain RAG
                rag_response = rag_chain.run({"context": context, "question": user_message})

                # Perform named entity recognition using BERT
                ner_results = ner_pipeline(user_message)

                # Perform sentiment analysis using RoBERTa
                sentiment_results = sentiment_pipeline(user_message)

                # Generate additional response using OpenAI
                openai_response = get_openai_response(user_message)

                st.write("RAG Response:", rag_response)
                st.write("NER Results:", ner_results)
                st.write("Sentiment Analysis:", sentiment_results)
                st.write("OpenAI Response:", openai_response)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
