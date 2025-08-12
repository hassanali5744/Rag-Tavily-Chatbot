-- Enhanced RAG Chatbot Database Schema
-- Run this to create all necessary tables

-- Documents table (stores uploaded documents)
CREATE TABLE IF NOT EXISTS documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content LONGTEXT NOT NULL,
    user_id INT DEFAULT 1,
    embedding JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_title (title)
);

-- Document chunks table (stores individual chunks for RAG)
CREATE TABLE IF NOT EXISTS document_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    INDEX idx_document_id (document_id),
    INDEX idx_chunk_index (chunk_index),
    UNIQUE KEY unique_chunk (document_id, chunk_index)
);

-- Chat messages table (stores conversation history)
CREATE TABLE IF NOT EXISTS chat_messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    role ENUM('user', 'assistant', 'system') NOT NULL,
    content TEXT NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_role (role),
    INDEX idx_created_at (created_at)
);

-- System prompts table (stores different system prompts)
CREATE TABLE IF NOT EXISTS system_prompts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL DEFAULT 'default',
    prompt TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name),
    INDEX idx_active (is_active)
);

-- Chat sessions table (for organizing conversations)
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    session_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
);

-- Session messages table (links messages to sessions)
CREATE TABLE IF NOT EXISTS session_messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    message_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_message_id (message_id)
);

-- Insert default system prompt
INSERT IGNORE INTO system_prompts (name, prompt) VALUES 
('default', 'You are a helpful AI assistant that can search through documents and the web to provide accurate information. Use the available tools to find relevant information and provide comprehensive answers.');

-- Sample data for testing
INSERT IGNORE INTO documents (title, content, user_id) VALUES 
('Sample Document', 'This is a sample document about machine learning and artificial intelligence. It contains information about various AI techniques and their applications.', 1);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_user_title ON documents(user_id, title);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_content ON document_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_messages_user_role ON chat_messages(user_id, role);
