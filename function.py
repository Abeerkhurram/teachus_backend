from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0,
                groq_api_key="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",
                model_name="mixtral-8x7b-32768")
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder="models")
    # Make sure HuggingFaceEmbeddings or SentenceTransformerEmbeddings is properly defined
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

def get_existing_vectorstore(filename):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = FAISS.load_local(filename, embeddings)

    return vectordb


def split_data(file_name):
    print(file_name)
    loader = PyPDFLoader(file_name)
    data = loader.load_and_split()

    return data

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     data_docs = text_splitter.split_text(text)
#     return data_docs

def retrieval_QA(vector_array):
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm = llm,
        retriever = vector_array.as_retriever(k=3)
    )
    return chain