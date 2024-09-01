from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader



# load the data
loader = CSVLoader(file_path="data/formatted_data.csv")
data = loader.load()

# split text into chunks
documents_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = documents_splitter.split_documents(data)

# embedding model
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',                                  
                                 model_kwargs={'device':'cpu'})


#Convert the Text Chunks into Embeddings and Create a Chroma Vector Store
vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory='./db3')
vectordb.persist()