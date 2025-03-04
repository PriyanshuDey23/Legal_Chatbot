from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup LLM for summarization
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

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