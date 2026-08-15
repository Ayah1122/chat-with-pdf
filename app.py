import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
import os

# --- Configuration: change this to your document's name/topic ---
DOCUMENT_TITLE = "الذكاء الاصطناعي"  # Shown in the page title/header
KNOWLEDGE_BASE_FILE = "knowledge_base.pdf"  # Must sit next to app.py in the repo

st.set_page_config(page_title=f"مساعد {DOCUMENT_TITLE}", page_icon="📚")
st.title(f"📚 مساعد {DOCUMENT_TITLE}")
st.caption("اسألني أي سؤال يتعلق بمحتوى هذا المستند فقط.")

# --- API key (Google Gemini - free tier, no credit card needed) ---
api_key = st.secrets.get("GOOGLE_API_KEY", None)
if not api_key:
    api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

if not api_key:
    st.info("Enter your free Gemini API key in the sidebar to start. Get one at aistudio.google.com/apikey")
    st.stop()

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

# --- Load the fixed knowledge-base PDF (bundled in the repo, not uploaded by users) ---
@st.cache_data(show_spinner="جاري تحميل المستند...")
def extract_text(path):
    reader = PdfReader(path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"\n\n--- Page {i+1} ---\n{page.extract_text() or ''}"
    return text

if not os.path.exists(KNOWLEDGE_BASE_FILE):
    st.error(
        f"لم يتم العثور على ملف '{KNOWLEDGE_BASE_FILE}'. "
        "تأكد من رفعه إلى مستودع GitHub بجانب app.py."
    )
    st.stop()

pdf_text = extract_text(KNOWLEDGE_BASE_FILE)
st.sidebar.success(f"تم تحميل المستند ({len(pdf_text.split())} كلمة)")

if len(pdf_text) > 400_000:
    st.sidebar.warning(
        "هذا المستند كبير — قد تكون الإجابات أبطأ. "
        "للمستندات الكبيرة جدًا، يُفضّل استخدام تقنية البحث المتجهي (v2)."
    )

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = (
        f"You are a specialized assistant that answers questions ONLY about "
        f"the document below, which covers the topic of '{DOCUMENT_TITLE}'. "
        "Rules:\n"
        "1. Answer ONLY using information found in the document content below.\n"
        "2. If a question is unrelated to the document's topic, politely say "
        "you can only answer questions about this specific document/topic, "
        "and do not attempt to answer from general knowledge.\n"
        "3. If the answer isn't in the document, say so clearly instead of guessing.\n"
        "4. Reply in the same language the user asked in (Arabic or English).\n\n"
        f"DOCUMENT CONTENT:\n{pdf_text}"
    )

    with st.chat_message("assistant"):
        with st.spinner("جاري التفكير..."):
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

