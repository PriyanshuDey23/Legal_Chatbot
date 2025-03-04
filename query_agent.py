from langchain_google_genai import ChatGoogleGenerativeAI
from vectordatabase import faiss_db # from vector store file
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Laod the Api Key
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

# Setup LLM 
llm_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=GOOGLE_API_KEY)


# Retrieve Docs
def retrieve_docs(query):
    return faiss_db.similarity_search(query)  # Similarity search

# Extract clean content from documents
def get_context(documents):
    return "\n\n".join([doc.page_content.strip() for doc in documents if doc.page_content])


# Define a clean custom prompt template
custom_prompt_template = """
Use the provided context to answer the user's question concisely and accurately.
- If the answer is not available in the context, state that you don't know.
- Do not generate information beyond the given context.

Question: {question}
Context:
{context}

Answer:
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



# Testing
# question="What are the transparency provisions outlined in the IFC regulations?"
# retrieved_docs=retrieve_docs(question)
# print(answer_query(documents=retrieved_docs, model=llm_model, query=question))