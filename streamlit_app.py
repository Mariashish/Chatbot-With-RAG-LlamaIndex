# Import necessary libraries
import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, get_response_synthesizer, Settings
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SentenceSplitter
import chromadb

# Add your own OPENAI API Key
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

# Display header and image
st.header("Chatbot with Retrieval Augmented Generation (RAG)")
st.image("Assistant_Image.png", width= 250) 

# Initialize chat message history if not present and give the proper Content
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Sono il tuo assistente virtuale personale! Chiedimi tutto ciò che desideri riguardo i PDF che mi sono stati caricati!"}
    ]

# Function to load data, caching to avoid repeated loading and give the proper System Prompt
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Sto caricando e indicizzando i documenti!"):
        reader = SimpleDirectoryReader(input_dir="Data", recursive=True)
        try: # Handle the errors
            docs = reader.load_data()
        except FileNotFoundError as e:
            st.error("The specified directory or files do not exist. Please check the file paths and try again.")
            st.error(f"Error details: {e}")
    

        # ChromaDB to save documents and not have to index everytime
        db = chromadb.PersistentClient(path="./chroma_db")
        # Create collection
        chroma_collection = db.get_or_create_collection("quickstart")
        # Assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Set up service context and index
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-turbo-preview", temperature=0.5, system_prompt="Sei un chatbot di assistente virtuale basato su Retrieval Augmented Generation (RAG) e il tuo compito è quello di rispondere alle domande che ti vengono fatte, basandoti sui PDF caricati. Presumi che tutte le domande siano relative ai documenti caricati. Mantieni le tue risposte tecniche e basate sui fatti, senza allucinazioni sulle caratteristiche."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index, storage_context

# Set chunk parameters
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=15)

# Create the index
index, storage_context = load_data()

# Set the retriever with the RAG parameters
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10, storage_context = storage_context, transformations = Settings.text_splitter
)

# Configure response synthesizer
response_synthesizer = get_response_synthesizer()

# Assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# Set up chat engine adn choose the Chat Mode, in this case condense_question
query_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})  # Save user input to chat history

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner(" "):
            response = query_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history

            # Display the sources for the message, including the score
            for node in response.source_nodes:
                print("-----")
                text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
                print(f"Text:\t {text_fmt} ...")
                print(f"Metadata:\t {node.node.metadata}")
                print(f"Score:\t {node.score:.3f}")
