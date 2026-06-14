import os
import shutil
from decouple import config # Importa as variáveis de ambiente
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader # O langchain possui vários loaders, além do PDF
from langchain_huggingface import HuggingFaceEmbeddings

# Busca as variáveis de ambiente
os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = config("HUGGINGFACE_API_KEY")

if __name__ == "__main__":
    # Define o caminho do diretório PDF para treinamento
    path = r"C:\Users\Nícolas Nasário\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Rag Aula\Rag\Data"

    # Cria uma lista para armazenar os documentos carregados
    all_docs = []

    for file_path in os.listdir(path):
        if file_path.endswith(".docx"): 
            print(f"Carregando documento: {file_path}")        
            # Faz o load do PDF para binário, memória do código
            # loader = PyPDFLoader(rf"C:\Users\nicol\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Rag Alexandre\Rag\Data\{file_path}")
            try:
                loader = Docx2txtLoader(rf"C:\Users\Nícolas Nasário\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Rag Aula\Rag\Data\{file_path}")
                # loader = PyPDFLoader(rf"C:\Users\nicol\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Rag Alexandre\Rag\Data\{file_path}")
                
                # Obtém os dados do documento
                docs = loader.load()

                # Adiciona os documentos carregados à lista
                all_docs.extend(docs)

                print(f"Documento carregado com sucesso: {file_path}")

            except Exception as e:
                print(f"Erro ao carregar o documento: {file_path}")
                print(e)

    # Splita a informação, dividindo em chunks e depois gera o overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Quantidade de tokens de tamanho
        chunk_overlap=200, # Quantidade de tokens de overlap para fazer a ligação das informações
    )

    # Agora gera o split dos documentos
    chunks = text_splitter.split_documents(
        documents=all_docs,
    )

    # Pasta onde vai ficar o banco de dados
    persist_directory = r"C:\Users\Nícolas Nasário\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Rag Aula\Pdf Data"

    # Verifica se chroma_data já existe como diretório ou arquivo
    if os.path.exists(persist_directory):
        if os.path.isdir(persist_directory):
            shutil.rmtree(persist_directory)  # Remove o diretório e seu conteúdo
        else:
            os.remove(persist_directory)  # Se for um arquivo, remove o arquivo

    # Busca o modelo de embedding
    embedding = HuggingFaceEmbeddings()

    # Cria o banco de dados a partir das informações obtidas
    vector_store = Chroma(
        embedding_function=embedding, # Passa o modelo do banco
        persist_directory=persist_directory, # Informa o diretório onde o banco vai ser criado
    )

    # Gera os documentos no banco de dados
    vector_store.add_documents(
        documents=chunks,
    )