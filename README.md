# 🤖 Enhanced RAG Chatbot with SQL Database

A powerful chatbot that combines **RAG (Retrieval-Augmented Generation)**, **LangChain**, **Tavily web search**, **MySQL database**, and **React frontend** to provide intelligent responses based on your documents and web information. This enhanced version uses a SQL database for persistent storage of documents, chat history, and user management.

## ✨ Features

- **📚 Document Upload & RAG**: Upload long text documents and search through them intelligently
- **🌐 Web Search**: Use Tavily to search the web for current information
- **🤖 LangChain Agent**: Powered by LangChain for intelligent tool selection and reasoning
- **💬 React Frontend**: Beautiful, responsive chat interface
- **🗄️ SQL Database**: MySQL database for persistent storage of documents and chat history
- **👥 Multi-User Support**: Support for multiple users with separate document collections
- **🔍 Vector Search**: Uses sentence transformers for semantic search
- **📊 Document Management**: Upload, view, and delete documents from the knowledge base
- **🔄 Graph Router**: Advanced routing system for complex queries

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  Python Backend │    │   External APIs │
│   (Port 5173)   │◄──►│   (Port 8000)   │◄──►│ Gemini + Tavily │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   MySQL Database│
                       │  + Vector Store │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 18+**
- **MySQL Server** (local or remote)
- **API Keys**: Gemini and Tavily (optional)

### 1. Database Setup

1. **Install MySQL Server** (if not already installed)
2. **Create a database:**
   ```sql
   CREATE DATABASE internship_chat1;
   ```
3. **The application will automatically create required tables on first run**

### 2. Install Dependencies

```bash
# Install Python dependencies
cd agent_python
pip install -r requirements.txt

# Install React dependencies
cd ../frontend
npm install
```

### 3. Set Up Environment Variables

Create a `.env` file in the `agent_python` directory:

```env
# Required: Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Tavily API Key (for web search)
TAVILY_API_KEY=your_tavily_api_key_here

# Database Configuration
DB_HOST=localhost
DB_USER=root
DB_PASS=123456
DB_NAME=internship_chat1

# Optional: Disable web search
DISABLE_WEB_SEARCH=false
```

**Get API Keys:**
- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Tavily**: [Tavily API](https://tavily.com/api-key) (free tier available)

### 4. Start the Services

**Terminal 1 - Start the Python Agent:**
```bash
cd agent_python
python start_sql_agent.py
```

**Terminal 2 - Start the React Frontend:**
```bash
cd frontend
npm run dev
```

### 5. Access the Application

- **Frontend**: http://localhost:5173
- **Agent API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 How to Use

### 1. Upload Documents

1. Click the **"+ Upload Document"** button in the sidebar
2. Enter a title for your document
3. Paste your long text content
4. Click **"Upload Document"**

The system will:
- Split your text into chunks
- Create embeddings for semantic search
- Store them in the database
- Associate them with your user ID

### 2. Chat with the Agent

Ask questions like:
- "What does the document say about machine learning?"
- "Search the web for the latest AI news"
- "Summarize the key points from my uploaded documents"
- "Compare information from different documents"

The agent will:
- Search through your uploaded documents using RAG
- Search the web using Tavily (if configured)
- Use the graph router for complex queries
- Combine information from both sources
- Provide a comprehensive response with sources

### 3. Manage Documents

- View all uploaded documents in the sidebar
- See how many chunks each document has been split into
- Delete documents you no longer need
- Documents are stored per user for privacy

## 🔧 API Endpoints

### Chat
```http
POST /chat
{
  "userId": 1,
  "message": "What is machine learning?",
  "system_prompt": "You are a helpful AI assistant..."
}
```

### Upload Document
```http
POST /upload_document
{
  "title": "Machine Learning Guide",
  "content": "Your long text content here...",
  "userId": 1
}
```

### List Documents
```http
GET /documents
```

### Delete Document
```http
DELETE /documents/{title}
```

### Health Check
```http
GET /health
```

## 🛠️ Technical Details

### RAG Implementation
- **Text Splitting**: RecursiveCharacterTextSplitter with 1000 char chunks
- **Embeddings**: HuggingFace sentence-transformers with fallback models
- **Vector Store**: In-memory with database persistence
- **Retrieval**: Top-3 most similar chunks

### Database Schema
- **documents**: Stores document metadata and content
- **document_chunks**: Stores text chunks with embeddings
- **chat_history**: Stores conversation history
- **users**: User management (future enhancement)

### LangChain Components
- **LLM**: Google Gemini 2.0 Flash
- **Agent Type**: Conversational React Description
- **Memory**: ConversationBufferMemory
- **Tools**: Document search + Tavily web search
- **Graph Router**: Advanced query routing

### Frontend Features
- **Real-time chat** with loading states
- **Document management** with upload/delete
- **Responsive design** for mobile/desktop
- **Source attribution** in responses
- **Modern UI** with animations

## 📁 Project Structure

```
├── agent_python/
│   ├── sql_enhanced_agent.py    # Main SQL-enhanced RAG agent
│   ├── graph_router.py          # Advanced query routing
│   ├── start_sql_agent.py       # Startup script
│   ├── requirements.txt         # Python dependencies
│   ├── mock_gemini.py           # Mock Gemini server
│   └── chroma_db/               # Vector store cache
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   └── App.css             # Styles
│   ├── package.json            # Frontend dependencies
│   └── README.md               # Frontend documentation
├── backend/                    # Node.js backend (optional)
└── README.md                   # This file
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check MySQL is running
   sudo systemctl status mysql
   
   # Verify database exists
   mysql -u root -p -e "SHOW DATABASES;"
   ```

2. **"Failed to fetch" errors**
   - Check if the Python agent is running on port 8000
   - Verify API keys are set correctly
   - Check browser console for CORS issues

3. **Missing dependencies**
   ```bash
   cd agent_python
   pip install -r requirements.txt
   ```

4. **Vector store errors**
   - Delete the `chroma_db` folder and restart
   - Check disk space for embeddings

5. **API key issues**
   - Verify keys are valid and have sufficient quota
   - Check `.env` file format (no spaces around `=`)

### Debug Mode

Run the startup script to check everything:
```bash
cd agent_python
python start_sql_agent.py
```

## 🚀 Deployment

### Local Development
```bash
# Start all services
cd agent_python && python start_sql_agent.py &
cd frontend && npm run dev &
```

### Production
- Use a proper WSGI server like Gunicorn
- Set up reverse proxy (nginx)
- Use environment variables for configuration
- Consider using Docker for containerization
- Set up proper MySQL production configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Run the diagnostic script
3. Check the logs for specific error messages
4. Open an issue with detailed information

## 🔮 Future Enhancements

- **User Authentication**: Secure login system
- **Document Sharing**: Share documents between users
- **Advanced Analytics**: Usage statistics and insights
- **Multi-language Support**: Support for multiple languages
- **File Upload**: Direct file upload support
- **Real-time Collaboration**: Multi-user chat sessions

---

**Happy chatting! 🤖✨**
