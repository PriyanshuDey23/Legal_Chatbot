
import streamlit as st
from query_agent import retrieve_docs, answer_query, llm_model
from summerize_agent import summarize_legal_text, llm
from langchain_core.messages import AIMessage, HumanMessage

def main():
    st.set_page_config(page_title="Legal AI Chatbot", layout="wide")
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
                summarized_response = summarize_legal_text(raw_response, llm)

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


# import streamlit as st
# from query_agent import retrieve_docs, answer_query, llm_model
# from summerize_agent import summarize_legal_text, llm

# def main():
#     st.set_page_config(page_title="Legal AI Chatbot", layout="wide")
#     st.title("⚖️ Legal AI Chatbot")
#     st.write("Get legal insights from trusted sources with AI-powered summarization.")
    
#     query = st.text_input("Ask a legal question:", )
#     if st.button("Get Answer"):
#         with st.spinner("Fetching legal information..."):
#             retrieved_docs = retrieve_docs(query)
#             raw_response = answer_query(documents=retrieved_docs, model=llm_model, query=query)
#             summarized_response = summarize_legal_text(raw_response, llm)
        
#         st.subheader("📜 Summarized Legal Answer ")
#         st.write(summarized_response)
        
#         with st.expander("🔍 View Extracted Legal Context"):
#             st.write(raw_response)

# if __name__ == "__main__":
#     main()