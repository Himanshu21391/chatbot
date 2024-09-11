import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import BertTokenizer, BertForTokenClassification, RobertaTokenizer, RobertaForTokenClassification, pipeline
import openai
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore  # Use the appropriate vector store class
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Define variables
docs_folder = os.path.join(os.path.dirname(__file__), 'docs')
chunk_size = 100000
chunk_overlap = 4
api_key = "d63e0bb3-ecc2-4e04-978f-3a4cbdbdf9fd"
index_name = "langchain-demo"
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
hf_api_token = "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"
openai_api_key = "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"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()
# Load and process documents
documents = []
for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(docs_folder, filename), encoding='utf-8')
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
docs = text_splitter.split_documents(documents)
vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
# Create an in-memory vector store (simplified)
doc_ids = [i for i in range(len(docs))]
doc_embeddings = [embeddings.embed(doc.text) for doc in docs]

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Embed documents using the correct method
doc_embeddings = embeddings.embed_documents([doc.text for doc in docs])  # This method works for embedding documents

def retrieve_similar_docs(query, top_k=5):
    # Embed query using the correct method
    query_embedding = embeddings.embed_query(query)
    
    # Calculate cosine similarities between the query embedding and document embeddings
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Retrieve top-k most similar documents
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

roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
roberta_model = RobertaForTokenClassification.from_pretrained(roberta_model_name)

# Set up NLP pipelines
ner_pipeline = pipeline("ner", model=bert_model, tokenizer=bert_tokenizer)
sentiment_pipeline = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer)

# Initialize OpenAI API
openai.api_key = openai_api_key

def get_openai_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use different engines like "text-davinci-003", "gpt-3.5-turbo", etc.
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "Please provide a message."}), 400

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

        response = {
            "rag_response": rag_response,
            "ner": ner_results,
            "sentiment": sentiment_results,
            "openai_response": openai_response
        }

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": f"An unexpected error occurred: {e}"}), 500

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)

