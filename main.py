import streamlit as st
import os
import sys

# Suppress the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Page configuration
st.set_page_config(
    page_title="Chat with Your Documents",
    page_icon="📚",
    layout="wide"
)

try:
    from utils import initialize_vector_store, add_documents_to_vector_store, Handle_pdf, clear_vector_store
    from Pages_.chatbot import YourDataChat
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Simple, clean CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        background-color: #f8f9fa;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize vector store and session state
initialize_vector_store()

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize processed files tracking
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Main header
st.title("📚 Chat with Your Documents")

# Sidebar for API Key and Upload
with st.sidebar:
    st.header("🔑 Settings")
    
    # API Key setup
    api_key_configured = 'OPEN_AI_KEY' in os.environ and os.environ['OPEN_AI_KEY']
    
    if not api_key_configured:
        api_key = st.text_input('OpenAI API Key', type='password', help="Enter your OpenAI API key to start chatting")
        if api_key and len(api_key) > 10:
            os.environ['OPEN_AI_KEY'] = api_key
            st.success("✅ API key set!")
            st.rerun()
        st.warning("⚠️ API key required for chat functionality")
    else:
        st.success("✅ API Key Configured")
        if st.button("Change API Key"):
            del os.environ['OPEN_AI_KEY']
            st.rerun()
    
    st.markdown("---")
    
    # Document upload section in sidebar
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with",
        key="pdf_uploader"
    )
    
    if uploaded_files:
        # Only process new files
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            with st.spinner("Processing documents..."):
                for file in new_files:
                    temp_path = None
                    try:
                        # Debug: File upload info
                        st.info(f"🔄 Processing file: {file.name} (Size: {file.size} bytes)")
                        
                        # Create temporary file
                        temp_path = f"temp_{file.name.replace(' ', '_')}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # Process PDF
                        pdf_handler = Handle_pdf(temp_path)
                        texts = pdf_handler.read_pdf()
                        
                        # Debug: PDF reading results
                        if texts:
                            st.success(f"✅ PDF loaded successfully! Found {len(texts)} text sections")
                            total_chars = sum(len(text) for text in texts)
                            st.info(f"📊 Total characters extracted: {total_chars:,}")
                            
                            # Process the text
                            success = add_documents_to_vector_store(texts, file.name)
                            
                            if success:
                                st.session_state.processed_files.add(file.name)
                                st.success(f"🎉 Successfully processed {file.name}")
                                # Use a targeted rerun to avoid clearing state unnecessarily
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to add {file.name} to vector store")
                        else:
                            st.error(f"❌ Failed to extract text from {file.name}")
                        
                        # Clean up temporary file
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")
                        # Clean up temporary file if it exists
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
    
    st.markdown("---")
    
    # Clear chat and data button
    if st.button("🗑️ Clear Chat & Stored Data"):
        clear_vector_store()
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Chat input
if not api_key_configured:
    # Disable chat input when no API key
    st.chat_input("Chat disabled - API key required", disabled=True)
else:
    if prompt := st.chat_input("Ask a question about your uploaded documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chatbot = YourDataChat()
                    
                    # Get context first
                    context = chatbot.get_context(prompt)
                    
                    # Generate response
                    response = chatbot.generate_openai_response(prompt, context)
                    
                    # Display response
                    st.write(response)
                    
                    # Display references if context was found
                    if context and context.get('returned_text') and len(context['returned_text']) > 0:
                        st.markdown("---")
                        st.markdown("**📚 References:**")
                        for i, (text, source) in enumerate(zip(context['returned_text'], context['source'])):
                            with st.expander(f"Reference {i+1} from {source}"):
                                st.write(text[:1000] + "..." if len(text) > 1000 else text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)


