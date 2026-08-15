import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# --- API key (Google Gemini - free tier, no credit card needed) ---
api_key = st.secrets.get("GOOGLE_API_KEY", None)
if not api_key:
    api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

if not api_key:
    st.info("Enter your free Gemini API key in the sidebar to start. Get one at aistudio.google.com/apikey")
    st.stop()

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

# --- PDF upload & text extraction (cached so re-runs don't re-parse) ---
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

@st.cache_data(show_spinner="Reading PDF...")
def extract_text(file_bytes):
    reader = PdfReader(file_bytes)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"\n\n--- Page {i+1} ---\n{page.extract_text() or ''}"
    return text

if uploaded_file:
    pdf_text = extract_text(uploaded_file)
    st.sidebar.success(f"Loaded {len(pdf_text.split())} words")

    # Simple guardrail: warn if the doc is very large for a single context window
    if len(pdf_text) > 400_000:  # ~100k tokens rough estimate
        st.sidebar.warning(
            "This PDF is large — answers may be slower or hit context limits. "
            "For big documents, a chunking/vector-search step (v2) works better."
        )
else:
    st.info("Upload a PDF in the sidebar to begin chatting.")
    st.stop()

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = (
        "You are a helpful assistant answering questions ONLY using the "
        "content of the PDF document provided below. If the answer isn't "
        "in the document, say so clearly instead of guessing.\n\n"
        f"DOCUMENT CONTENT:\n{pdf_text}"
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            gemini_contents = [
                types.Content(
                    role="model" if m["role"] == "assistant" else "user",
                    parts=[types.Part(text=m["content"])],
                )
                for m in st.session_state.messages
            ]
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=gemini_contents,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
            answer = response.text
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
