# 🤖 Chat with Your Documents - RAG Implementation

> **Related Article**: [Bring Your Data to Life: Creating a Chatbot with LLM, LangChain, Vector DB Locally on Docker](https://medium.com/@mutazyounes/bring-your-data-to-life-creating-a-chatbot-with-llm-langchain-vector-db-locally-on-docker-ed647e546f85)

A modern, powerful document chat application that implements **Retrieval-Augmented Generation (RAG)** using OpenAI's GPT models, LangChain, and FAISS vector database. Upload your PDF documents and chat with them using advanced AI capabilities.

## 🎯 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines the power of large language models with external knowledge retrieval. Instead of relying solely on the model's pre-trained knowledge, RAG:

1. **Retrieves** relevant information from your documents
2. **Augments** the LLM's context with this information
3. **Generates** accurate, contextual responses

This approach ensures that the AI can provide specific, factual answers based on your actual documents rather than generic responses.

## ✨ Features

- **📁 Document Upload**: Upload multiple PDF files with drag-and-drop interface
- **🔍 Vector Search**: Advanced FAISS-based similarity search for accurate document retrieval
- **🤖 AI Chat**: Chat with your documents using OpenAI's GPT-4o-mini
- **📊 Real-time Statistics**: View document and chunk statistics
- **🎨 Modern UI**: Clean, responsive Streamlit interface
- **⚙️ Easy Configuration**: Simple API key setup
- **🐳 Docker Support**: Run locally or in containers

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Chunking  │───▶│  Vector Store   │
│                 │    │                 │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Vector Search  │◀───│   Embeddings    │
│                 │    │                 │    │ (BGE Model)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Response   │◀───│  GPT-4o-mini    │◀───│  Context +      │
│                 │    │                 │    │  Query          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Docker (optional, for containerized deployment)

### Method 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chat-with-your-data
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # Or create a .env file with: OPENAI_API_KEY=your-api-key-here
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

### Method 2: Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**
   ```bash
   docker build -t chat-with-documents .
   docker run -p 8501:8501 -e OPENAI_API_KEY=your-key chat-with-documents
   ```

## 📖 How to Use

### 1. Set Up API Key
- In the sidebar, enter your OpenAI API key
- The application will validate and store your key securely

### 2. Upload Documents
- Click "Choose files" in the upload section
- Select one or more PDF files
- Click "Process Documents" to add them to your knowledge base
- Watch the real-time processing statistics

### 3. Start Chatting
- Once documents are uploaded, the chat interface will appear
- Ask questions about your documents
- The AI will search through your documents and provide relevant answers with source citations

### 4. View Statistics
- Check the sidebar for real-time statistics
- See how many documents and text chunks you have processed

## 🛠️ Technical Details

### Core Components

- **`main.py`**: Main Streamlit application entry point
- **`utils.py`**: Core utilities for vector operations and PDF processing
- **`Pages_/chatbot.py`**: AI chat functionality with RAG implementation
- **`Pages_/knowledge_base.py`**: Document processing and chunking
- **`Pages_/vector_manager.py`**: FAISS vector store management
- **`config.py`**: Configuration settings

### Key Technologies

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python with OpenAI API integration
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: BAAI/bge-base-en sentence transformers
- **LLM**: OpenAI GPT-4o-mini for response generation
- **Document Processing**: PyPDF2 for PDF text extraction

### File Structure
```
chat-with-your-data/
├── main.py                 # Main Streamlit application
├── utils.py               # Core utilities and vector operations
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── .gitignore            # Git ignore patterns
├── README.md             # This file
├── Pages_/               # Application modules
│   ├── chatbot.py        # RAG chat functionality
│   ├── knowledge_base.py # Document processing
│   └── vector_manager.py # Vector store management
└── static/               # Static assets
```

## 🎨 Design Features

### Modern UI Elements
- **Clean Layout**: Organized sections for upload, chat, and settings
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Real-time Feedback**: Live processing statistics and status updates
- **Intuitive Navigation**: Clear sidebar with all controls

### User Experience
- **Drag-and-Drop Upload**: Easy document upload interface
- **Progress Indicators**: Visual feedback during document processing
- **Error Handling**: Clear error messages and recovery options
- **Session Persistence**: Maintains chat history and document state

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-4o-mini)
- `EMBEDDING_MODEL`: Embedding model (default: BAAI/bge-base-en)

### Model Settings
- **Embedding Model**: BAAI/bge-base-en (optimized for English)
- **Chat Model**: GPT-4o-mini (latest OpenAI model)
- **Vector Search**: FAISS with cosine similarity
- **Chunk Size**: 1000 characters with 200 character overlap

## 📝 Usage Examples

### Example Questions
- "What are the main topics discussed in the documents?"
- "Can you summarize the key findings from the research papers?"
- "What does the document say about machine learning applications?"
- "Find information about specific methodologies mentioned in the documents"
- "Compare the approaches discussed in different documents"

### Document Types Supported
- PDF files (text-based and image-based with OCR)
- Multiple languages (with appropriate embedding models)
- Large documents (automatically chunked for processing)

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your OpenAI API key is valid and has sufficient credits
   - Check that the key is properly set in environment variables
   - Verify the key format (starts with `sk-`)

2. **Document Processing Issues**
   - Ensure PDF files are not corrupted or password-protected
   - For image-based PDFs, consider using OCR tools
   - Large files may take longer to process

3. **Memory Issues**
   - Large documents are automatically chunked
   - Consider processing documents in smaller batches
   - Monitor system memory usage during processing

4. **Vector Store Issues**
   - The vector store is automatically created in the `vector_store/` directory
   - Ensure write permissions in the project directory
   - Clear the vector store if you encounter corruption issues

### Performance Tips
- Use smaller PDF files for faster processing
- Clear chat history periodically to free memory
- Save vector store regularly to preserve processed documents
- Consider using GPU acceleration for embedding generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT API and advancing LLM technology
- **LangChain** for the excellent framework for building LLM applications
- **FAISS** for efficient vector similarity search
- **Sentence Transformers** for high-quality embeddings
- **Streamlit** for the amazing web framework
- **Medium Community** for the inspiration and knowledge sharing

## 🔗 Related Resources

- [Medium Article: Bring Your Data to Life](https://medium.com/@mutazyounes/bring-your-data-to-life-creating-a-chatbot-with-llm-langchain-vector-db-locally-on-docker-ed647e546f85)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Made with ❤️ for document intelligence and RAG applications**

*This project demonstrates the power of Retrieval-Augmented Generation (RAG) in creating intelligent document chatbots that can provide accurate, contextual responses based on your specific documents.*
