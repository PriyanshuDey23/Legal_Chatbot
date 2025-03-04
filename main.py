import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

#  Streamlit Config
st.set_page_config(page_title="Legal AI Chatbot", layout="wide")

#  Load API Key
GOOGLE_API_KEY=st.secrets["GOOGLE_API_KEY"]

if not GOOGLE_API_KEY:
    st.error("⚠️ Google API Key is missing! Add it to Streamlit secrets.")
    st.stop()

#  Directories & Document Loading
PDFS_DIRECTORY = "Data/"
os.makedirs(PDFS_DIRECTORY, exist_ok=True)

def load_documents(folder_path):
    """Loads PDF documents dynamically from the folder."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        return []
    
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(folder_path, pdf))
        documents.extend(loader.load())
    return documents

#  Chunking Documents
def create_chunks(documents):
    """Splits documents into chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)

#  Initialize FAISS Index (Lazy Loading)
if "faiss_db" not in st.session_state:
    documents = load_documents(PDFS_DIRECTORY)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if documents:
        with st.spinner("🔍 Creating FAISS Index..."):
            text_chunks = create_chunks(documents)
            st.session_state.faiss_db = FAISS.from_documents(text_chunks, embedding_model)
    else:
        st.session_state.faiss_db = None  # No documents available

#  LLM Initialization
llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
llm_summarizer = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)


#  Retrieve Documents
def retrieve_docs(query):
    """Fetch relevant documents based on similarity search."""
    if not st.session_state.faiss_db:
        return []
    
    retrieved_docs = st.session_state.faiss_db.similarity_search(query, k=5)
    return retrieved_docs or []

#  Extract Context
def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


#  Answer Query
custom_prompt_template = """
You are a legal AI assistant. Answer the user's question based on the provided legal context.
- Use clear and concise language.
- Do NOT make up information. If the answer isn't in the context, state that.

**Question:** {question}  
**Context:**  
{context}  

**Answer:**  
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    if not context:
        return "I'm sorry, but I couldn't find relevant information in the provided context."
    
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model  # Pipeline execution
    response = chain.invoke({"question": query, "context": context})
    
    # Extract and return only the text response, handling potential metadata issues
    return response.content if hasattr(response, 'content') else str(response)


# Summarization prompt template
summarization_prompt_template = """
You are a legal summarization expert. Your task is to simplify complex legal text while ensuring accuracy.
- Preserve all essential details.
- Use plain, easy-to-understand language.
- Avoid unnecessary legal jargon unless required.

Original Legal Text:
{legal_text}

Simplified Summary:
"""

def summarize_legal_text(legal_text, model):
    if not legal_text.strip():
        return "No legal text provided for summarization."
    
    prompt = ChatPromptTemplate.from_template(summarization_prompt_template)
    chain = prompt | model  # Pipeline execution
    response = chain.invoke({"legal_text": legal_text})
    
    return response.content if hasattr(response, 'content') else str(response)



#  Streamlit UI
def main():
    st.title("⚖️ Legal AI Chatbot")
    st.write("Get legal insights from trusted sources with AI-powered summarization.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm a Legal AI Assistant. Ask me anything about legal topics.")
        ]

    # Display chat messages
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    # Handle user input
    user_query = st.chat_input("Type a legal question...")
    if user_query and user_query.strip():
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            with st.spinner("Fetching legal information..."):
                retrieved_docs = retrieve_docs(user_query)
                raw_response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
                summarized_response = summarize_legal_text(raw_response, llm_summarizer)

            # Display summarized answer
            st.subheader("📜 Summarized Legal Answer")
            st.write(summarized_response)

            # Expandable section for extracted legal text
            with st.expander("🔍 View Extracted Legal Context"):
                st.write(raw_response)

            # Store AI response in chat history
            st.session_state.chat_history.append(AIMessage(content=summarized_response))

if __name__ == "__main__":
    main()

