from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import pinecone
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
import os


class ChatBot:
    def __init__(self):
        # Loading environment variables from .env file
        load_dotenv()

        # Loading text data from a file using TextLoader
        loader = TextLoader('./dataset.txt', encoding="utf8")
        documents = loader.load()

        # Splitting documents into chunks using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Initializing Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings()

        # Initializing Pinecone for efficient vector search
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment='gcp-starter'
        )

        index_name = "rag-chatbot"

        # Checking if index exists in Pinecone, creating if not exists
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(index_name, embeddings)

        # Initializing Hugging Face endpoint with specified parameters
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.1, "top_k": 50, "max_new_tokens": 2048},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Defining a prompt template for the chatbot's responses
        template = """
        "Use the following pieces of context to answer the question at the end with human readable answer as a paragraph."
        "Please do not use data outside the context to answer any questions."
        "If the answer is not in the given context, just say that you don't have enough context."

        Context: {context}
        Question: {question}
        Answer: 
        """

        # Instantiating a PromptTemplate based on the defined template
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Defining the processing chain for the chatbot
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def get_response(self, user_input):
        return self.rag_chain.invoke(user_input)
