import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page configuration
st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks, api_key):
    """Create FAISS vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversational_chain(api_key):
    """Create conversational chain for question answering"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "The answer is not available in the uploaded documents."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        google_api_key=api_key
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create the chain using LCEL (LangChain Expression Language)
    chain = (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain

def process_user_question(user_question, api_key):
    """Process user question and generate response"""
    if st.session_state.vector_store is None:
        st.error("Please upload and process documents first!")
        return None
    
    # Retrieve relevant documents
    docs = st.session_state.vector_store.similarity_search(user_question, k=3)
    
    # Get conversational chain
    chain = get_conversational_chain(api_key)
    
    # Generate response
    response = chain.invoke({"docs": docs, "question": user_question})
    
    return response

def main():
    st.title("📚 RAG Document Q&A with Google Gemini")
    st.markdown("Upload PDF documents and ask questions about their content!")
    
    # Sidebar for API key and document upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter Google Gemini API Key",
            type="password",
            help="Get your API key from https://aistudio.google.com/app/apikey"
        )
        
        st.markdown("---")
        st.header("📄 Upload Documents")
        
        # File uploader
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        # Process button
        if st.button("Process Documents", type="primary"):
            if not api_key:
                st.error("Please enter your Google Gemini API key!")
            elif not pdf_docs:
                st.error("Please upload at least one PDF file!")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        # Extract text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        # Split into chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Create vector store
                        st.session_state.vector_store = create_vector_store(text_chunks, api_key)
                        
                        st.success(f"✅ Processed {len(pdf_docs)} document(s) successfully!")
                        st.info(f"Created {len(text_chunks)} text chunks for retrieval")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
        
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**Q{i+1}:** {question}")
                st.markdown(f"**A{i+1}:** {answer}")
                st.markdown("---")
        
        # Question input
        user_question = st.text_input(
            "Ask a question about your documents:",
            key="question_input",
            placeholder="e.g., What is the main topic of the document?"
        )
        
        if st.button("Get Answer", type="primary"):
            if not api_key:
                st.error("Please enter your Google Gemini API key in the sidebar!")
            elif not user_question:
                st.warning("Please enter a question!")
            elif st.session_state.vector_store is None:
                st.error("Please upload and process documents first!")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        answer = process_user_question(user_question, api_key)
                        if answer:
                            st.session_state.chat_history.append((user_question, answer))
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
    
    with col2:
        st.header("ℹ️ Info")
        st.info(
            """
            **How to use:**
            1. Enter your Google Gemini API key
            2. Upload PDF documents
            3. Click 'Process Documents'
            4. Ask questions about the content
            
            **Features:**
            - Multi-document support
            - Context-aware answers
            - Chat history
            - Semantic search
            """
        )
        
        if st.session_state.vector_store:
            st.success("✅ Documents loaded and ready!")
        else:
            st.warning("⚠️ No documents loaded yet")

if __name__ == "__main__":
    main()
