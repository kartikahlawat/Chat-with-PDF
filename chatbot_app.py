import streamlit as st
from ingest import ingest_pdf, search_similarity
from constants import OPENAI_API_KEY
import openai

st.set_page_config(page_title="Chat with PDF", layout="wide")

st.title("Chat with PDF 🤖📄")
st.markdown("Upload a PDF and interact with it using AI!")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for API key and PDF upload
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        value=OPENAI_API_KEY if OPENAI_API_KEY else "",
        type="password",
        help="You can get your key from https://platform.openai.com/account/api-keys"
    )
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Ingest PDF and store embeddings
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            docs, embeddings = ingest_pdf(uploaded_file)
            st.session_state["docs"] = docs
            st.session_state["embeddings"] = embeddings
            st.success("PDF processed and ready!")
        except Exception as e:
            st.error(f"Error: {e}")

# Input box for user question
prompt = st.text_input("Ask a question about your PDF:")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# Handle user prompt
if prompt and uploaded_file is not None and "embeddings" in st.session_state:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Find relevant context from PDF
    relevant_chunks = search_similarity(
        prompt, st.session_state["docs"], st.session_state["embeddings"]
    )
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])

    # Combine context and prompt
    full_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {prompt}\n"
        "Answer:"
    )

    # Query OpenAI
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        openai.api_key = api_key
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that answers questions about PDF documents."},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            answer = response["choices"][0]["message"]["content"]
        except Exception as e:
            answer = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f"**Bot:** {answer}")