import os
from decouple import config # Busca a variável de ambiente
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq # Busca os modelos do groq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

# Comunica com o groq
os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")

# Classe do bot
class AIBot():    
    # Inicializador
    def __init__(self):
        # Cria o modelo do Groq
        self.__chat = ChatGroq(model="llama-3.3-70b-versatile")
        # Cria um retriever, uma instância do banco de dadosd
        self.__retriever = self.__build_retriever()

    # Localiza o banco de dados e busca as informações conforme o prompt
    def __build_retriever(self):
        # Informa o local do banco de dados
        persist_directory = r"C:\Users\Nícolas Nasário\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Rag Aula\Pdf Data"
        # Busca as informações relevantes
        embedding = HuggingFaceEmbeddings()

        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )

        return vector_store.as_retriever(
            search_kwargs={"k": 2}, # Busca até X resultados no máximo
        )

    def __build_messages(self, history_messages, question):
        # Garantir que history_messages não seja None
        if history_messages is None:
            history_messages = []

        messages = []

        # Construir a lista de mensagens a partir do histórico
        for message in history_messages:
            # Verifica se a chave 'fromMe' está presente no dicionário
            message_class = HumanMessage if message.get("fromMe") else AIMessage
            messages.append(message_class(content=message.get("body", "")))

        # Adiciona a mensagem atual (pergunta)
        messages.append(HumanMessage(content=question))

        return messages

    # Recebe a mensagem do usuário e processa a partir de um prompt
    def invoke(self, question):
        SYSTEM_TEMPLATE = """
        Você é um assistente pessoal de um aluno do curso de Análise e desenvolvimento de sistemas. Você deve usar o seu conhecimento para responder as perguntas do contexto.
        Não invente nada, você deve sempre buscar dados no seu conhecimento.

        <context>
        {context}
        </context>
        """

        # Busca as informações sobre a questão do usuário. Faz um tipo de requisição (como se fosse uma query)
        docs = self.__retriever.invoke(question)

        # Forma um template para a IA, usando o langchain
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE, # Passa a mensagem de sistema, que é o que foi definido anteriormente (prompt enviado para a IA)
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        document_chain = create_stuff_documents_chain(self.__chat, question_answering_prompt)

        response = document_chain.invoke(
            {
                "context": docs, # Troca a variável {context} do template pelos documentos que subiram no banco vetorizado
                "messages": self.__build_messages(question=question, history_messages=None), # Passa em messages o histórico de mensagens com o usuário
            }
        )

        return response