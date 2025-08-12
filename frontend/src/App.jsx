import { useState, useEffect, useRef } from "react";
import "./App.css";

export default function App() {
  const [text, setText] = useState("");
  const [msgs, setMsgs] = useState([{ role: 'system', content: 'Welcome to the Enhanced RAG Chatbot! I can search through your documents and the web to provide accurate information.' }]);
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [uploadTitle, setUploadTitle] = useState("");
  const [uploadContent, setUploadContent] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [msgs]);

  useEffect(() => {
    loadDocuments();
  }, []);

  async function loadDocuments() {
    try {
      const response = await fetch("http://localhost:8000/documents");
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  }

  async function send() {
    if (!text.trim()) return;
    
    const m = { role: 'user', content: text };
    setMsgs(s => [...s, m]);
    setText('');
    setLoading(true);
    
    try {
      console.log('Sending message:', m.content);
      const r = await fetch("http://localhost:8000/chat", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          userId: 1, 
          message: m.content,
          system_prompt: "You are a helpful AI assistant that can search through documents and the web to provide accurate information."
        })
      });
      
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}: ${r.statusText}`);
      }
      
      const j = await r.json();
      console.log('Received response:', j);
      
      // Format response with sources
      let responseContent = j.reply || '(no reply)';
      console.log('Response content:', responseContent);
      
      if (j.sources && j.sources.length > 0) {
        responseContent += '\n\n📚 Sources:';
        j.sources.forEach((source, index) => {
          responseContent += `\n${index + 1}. ${source.title}: ${source.content}`;
        });
      }
      
      console.log('Final response content:', responseContent);
      setMsgs(s => [...s, { role: 'assistant', content: responseContent }]);
    } catch (err) {
      console.error('Error in send:', err);
      setMsgs(s => [...s, { role: 'assistant', content: 'Error: ' + String(err) }]);
    } finally {
      setLoading(false);
    }
  }

  async function uploadDocument() {
    if (!uploadTitle.trim() || !uploadContent.trim()) {
      alert("Please provide both title and content");
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/upload_document", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: uploadTitle,
          content: uploadContent,
          userId: 1
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      alert(`Document uploaded successfully! ${data.message}`);
      
      // Reset form
      setUploadTitle("");
      setUploadContent("");
      setShowUpload(false);
      
      // Reload documents
      loadDocuments();
    } catch (err) {
      alert("Error uploading document: " + String(err));
    }
  }

  async function deleteDocument(title) {
    if (!confirm(`Are you sure you want to delete "${title}"?`)) return;

    try {
      const response = await fetch(`http://localhost:8000/documents/${encodeURIComponent(title)}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      alert("Document deleted successfully!");
      loadDocuments();
    } catch (err) {
      alert("Error deleting document: " + String(err));
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="app">
      <div className="header">
        <h1>🤖 Enhanced RAG Chatbot</h1>
        <p>Powered by LangChain, RAG, and Tavily Search</p>
      </div>

      <div className="main-container">
        <div className="sidebar">
          <div className="sidebar-section">
            <h3>📚 Knowledge Base</h3>
            <button 
              className="upload-btn"
              onClick={() => setShowUpload(!showUpload)}
            >
              {showUpload ? 'Cancel' : '+ Upload Document'}
            </button>
            
            {showUpload && (
              <div className="upload-form">
                <input
                  type="text"
                  placeholder="Document Title"
                  value={uploadTitle}
                  onChange={(e) => setUploadTitle(e.target.value)}
                />
                <textarea
                  placeholder="Document Content (paste your long text here)..."
                  value={uploadContent}
                  onChange={(e) => setUploadContent(e.target.value)}
                  rows={8}
                />
                <button onClick={uploadDocument} className="upload-submit">
                  Upload Document
                </button>
              </div>
            )}
          </div>

          <div className="sidebar-section">
            <h3>📖 Documents ({documents.length})</h3>
            <div className="documents-list">
              {documents.map((doc, index) => (
                <div key={index} className="document-item">
                  <div className="document-info">
                    <strong>{doc.title}</strong>
                    <span>{doc.chunks} chunks</span>
                  </div>
                  <button 
                    onClick={() => deleteDocument(doc.title)}
                    className="delete-btn"
                  >
                    🗑️
                  </button>
                </div>
              ))}
              {documents.length === 0 && (
                <p className="no-docs">No documents uploaded yet</p>
              )}
            </div>
          </div>
        </div>

        <div className="chat-container">
          <div className="chat-messages">
            {msgs.map((m, i) => (
              <div key={i} className={`message ${m.role}`}>
                <div className="message-header">
                  <span className="role-badge">{m.role}</span>
                </div>
                <div className="message-content">
                  {m.content}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message assistant">
                <div className="message-header">
                  <span className="role-badge">assistant</span>
                </div>
                <div className="message-content">
                  <div className="loading">Thinking...</div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything... I can search through your documents and the web!"
              rows={3}
            />
            <button onClick={send} disabled={loading || !text.trim()}>
              {loading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}