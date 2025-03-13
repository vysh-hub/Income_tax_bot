import streamlit as st
import faiss
import openai
import numpy as np
import json
import os

# Load FAISS Index & Finance Bill JSON

try:
    index = faiss.read_index("updated_finance_bill_index.faiss")
    with open("merged_data.json", "r", encoding="utf-8") as f:
        finance_data = json.load(f)
except Exception as e:
    st.error(f"❌ ERROR: Failed to load data: {e}")
    st.stop()

#  Set OpenAI API Key (Make sure it's set in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("❌ ERROR: OpenAI API Key is missing. Set it in your environment!")
    st.stop()

#  Function to Convert Query to Embedding
def get_query_embedding(query):
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[query]
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        return embedding
    except Exception as e:
        st.error(f"❌ ERROR: Failed to generate embedding. {e}")
        return None

#  Function to Summarize & Explain Response Using GPT-3.5
def summarize_with_gpt3(raw_text, user_query):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a tax expert helping users understand finance bills in simple terms."},
                {"role": "user", "content": f"The user asked: {user_query}\n\nThe retrieved finance bill section is:\n{raw_text}\n\nExplain this in a simple and easy-to-understand manner."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ ERROR: Failed to summarize response with GPT-3.5.\nError: {e}")
        return raw_text

#  Function to Search FAISS & Summarize with GPT-3.5
def search_finance_bill(query):
    query_embedding = get_query_embedding(query)
    
    if query_embedding is None:
        return "❌ ERROR: Could not generate an embedding for the query."

    distances, indices = index.search(query_embedding, k=1)

   
    
    # Check if FAISS gave a bad match
    if indices[0][0] == -1 or distances[0][0] > 0.4:  # Adjust threshold based on FAISS output

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a tax expert helping users understand tax-related queries."},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ ERROR: OpenAI API call failed. {e}"

    # FAISS returned a match, process it
    best_match_index = indices[0][0]
    if best_match_index >= len(finance_data):
        return "❌ ERROR: Retrieved index is out of range."

    retrieved_text = finance_data[best_match_index]["content"]
    return summarize_with_gpt3(retrieved_text, query)


#  Streamlit UI
st.title("💰 TaxBot - Ask Your Tax Questions")
st.write("Enter a tax-related question, and I'll find the answer for you!")

if 'messeges' not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

#  User Input
user_query = st.text_input("Ask a question about taxes:")

if st.button("Ask TaxBot"):
    if user_query:
        st.session_state.messages.append({"role":"assistant","content":user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.spinner("Fetching response..."):
            response = search_finance_bill(user_query)
            st.session_state.messages.append({"role":"assistant","content":response})
        with st.chat_message("user"):
            st.markdown(response)           
            
    else:
        st.warning("⚠ Please enter a question.")


