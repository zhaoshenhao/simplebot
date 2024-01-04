from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

vdb_dir = 'vdb'
loader = PyPDFLoader('about_running_injuries.pdf')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print("Save to database %s." % vdb_dir)
vectordb  = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(), persist_directory=vdb_dir)
vectordb.persist()
print("Done")
