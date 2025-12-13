# # app.py - uses Groq (free & blazing fast)
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# load_dotenv()

# st.set_page_config(page_title="AskHR", page_icon="Robot")
# st.title("AskHR – Your AI HR Assistant (Live Demo)")

# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about leave, benefits, payroll, WFH..."}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# if prompt := st.chat_input("Ask your HR question..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = PineconeVectorStore(index_name="hr-bot", embedding=embeddings)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#     template = """You are AskHR, a helpful HR assistant. Answer ONLY from the context.
#     Context: {context}
#     Question: {question}
#     Answer:"""

#     prompt_template = PromptTemplate.from_template(template)
#     llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt_template
#         | llm
#         | StrOutputParser()
#     )

#     with st.spinner("Thinking..."):
#         response = chain.invoke(prompt)

#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.chat_message("assistant").write(response)

# app.py — FINAL CLEAN ENTERPRISE VERSION (No Sidebar) — Dec 10, 2025
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="AskHR – Enterprise AI HR Assistant", page_icon="Robot", layout="centered")


st.markdown("""
<style>
    .main {background: #000000; color: #e2e8f0;}
    .stChatMessage {margin: 20px 0; padding: 16px 20px; border-radius: 20px; max-width: 80%;}
    .user {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;}
    .assistant {background: #1a1a1a; border: 1px solid #333; color: #e2e8f0;}
    .source {font-size: 0.75rem; color: #94a3b8; background: #27272a; padding: 5px 12px; border-radius: 12px; margin-top: 8px;}
    .title {
        font-size: 4.5rem; font-weight: 900; text-align: center; margin: 2rem 0;
        background: linear-gradient(90deg, #667eea, #764ba2, #f783ac, #667eea);
        background-size: 200%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: gradient 8s ease infinite;
    }
    @keyframes gradient {0%,100%{background-position:0% 50%} 50%{background-position:100% 50%}}
    .subtitle {text-align: center; color: #94a3b8; font-size: 1.4rem; margin-bottom: 3rem;}
    .chat-container {max-width: 900px; margin: 0 auto;}
</style>
""", unsafe_allow_html=True)

# ── Title ──
st.markdown("<h1 class='title'>AskHR</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enterprise RAG-Powered HR Assistant • 82-Page Policy Knowledge Base</p>", unsafe_allow_html=True)

# # Clear Chat Button 
# col1, col2 = st.columns([6, 1])
# with col2:
#     if st.button("Clear", type="primary"):
#         st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about HR policies..."}]
#         st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm AskHR — your 24×7 AI assistant for leave, benefits, payroll, WFH, and more.\nAsk me anything!"}]

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            for src in msg["sources"]:
                st.markdown(f"<span class='source'>Source: {src}</span>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── User Input ──
if prompt := st.chat_input("Ask your HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.spinner("Searching policies..."):
        # 1. Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
        # 2. Connect to Pinecone (v5)
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("hr-bot")
    
        # 3. Create vectorstore from existing index
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
    
        # 4. Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


        template = """You are AskHR, a professional and accurate HR assistant.
        Answer ONLY from the context. Be concise and friendly.
        Context: {context}
        Question: {question}
        Answer:"""

        prompt_template = PromptTemplate.from_template(template)
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())
        response = chain.invoke(prompt)
        docs = retriever.invoke(prompt)
        sources = list(set([doc.metadata.get("source", "Policy").split("/")[-1] for doc in docs]))

    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})

    st.rerun()
