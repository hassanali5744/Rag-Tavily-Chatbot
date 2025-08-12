#!/usr/bin/env python3
"""
Startup script for SQL-based Enhanced RAG Chatbot
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_python_dependencies():
    """Check if required Python packages are installed"""
    print("🔍 Checking Python Dependencies...")
    print("=" * 50)
    
    required_packages = [
        'fastapi',
        'uvicorn', 
        'requests',
        'python-dotenv',
        'langchain',
        'langchain-google-genai',
        'langchain-community',
        'langchain-core',
        'tavily-python',
        'sentence-transformers',
        'pydantic',
        'python-multipart',
        'mysql-connector-python',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All Python dependencies are installed!")
    return True

def check_environment_variables():
    """Check if required environment variables are set"""
    print("\n🔍 Checking Environment Variables...")
    print("=" * 50)
    
    required_vars = {
        'GEMINI_API_KEY': 'Gemini API key for LLM',
        'TAVILY_API_KEY': 'Tavily API key for web search',
        'DB_HOST': 'Database host (default: localhost)',
        'DB_USER': 'Database user (default: root)',
        'DB_PASS': 'Database password (default: 123456)',
        'DB_NAME': 'Database name (default: internship_chat)'
    }
    
    all_set = True
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: SET")
        else:
            print(f"❌ {var}: NOT SET - {description}")
            all_set = False
    
    if not all_set:
        print("\n⚠️ Some environment variables are missing.")
        print("Please check your .env file")
        return False
    
    print("\n✅ All environment variables are set!")
    return True

def check_database_connection():
    """Test database connection"""
    print("\n🔍 Testing Database Connection...")
    print("=" * 50)
    
    try:
        import mysql.connector
        from mysql.connector import Error
        
        DB_HOST = os.getenv('DB_HOST', 'localhost')
        DB_USER = os.getenv('DB_USER', 'root')
        DB_PASS = os.getenv('DB_PASS', '123456')
        DB_NAME = os.getenv('DB_NAME', 'internship_chat')
        
        print(f"Connecting to: {DB_HOST}/{DB_NAME}")
        
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            print("✅ Database connection successful!")
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Error as e:
        print(f"❌ Database connection error: {e}")
        print("\n💡 Make sure:")
        print("1. MySQL server is running")
        print("2. Database 'internship_chat' exists")
        print("3. User has proper permissions")
        print("4. Check your .env file for correct credentials")
        return False
    except ImportError:
        print("❌ mysql-connector-python not installed")
        print("Run: pip install mysql-connector-python")
        return False

def check_database_tables():
    """Check if required database tables exist"""
    print("\n🔍 Checking Database Tables...")
    print("=" * 50)
    
    try:
        import mysql.connector
        from mysql.connector import Error
        
        DB_HOST = os.getenv('DB_HOST', 'localhost')
        DB_USER = os.getenv('DB_USER', 'root')
        DB_PASS = os.getenv('DB_PASS', '123456')
        DB_NAME = os.getenv('DB_NAME', 'internship_chat')
        
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        
        cursor = connection.cursor()
        
        required_tables = [
            'documents',
            'document_chunks', 
            'chat_messages',
            'system_prompts',
            'chat_sessions',
            'session_messages'
        ]
        
        all_tables_exist = True
        
        for table in required_tables:
            cursor.execute(f"SHOW TABLES LIKE '{table}'")
            result = cursor.fetchone()
            if result:
                print(f"✅ {table}")
            else:
                print(f"❌ {table}")
                all_tables_exist = False
        
        cursor.close()
        connection.close()
        
        if not all_tables_exist:
            print("\n⚠️ Some tables are missing.")
            print("Run the SQL schema: mysql -u root -p internship_chat < database_schema.sql")
            return False
        
        print("\n✅ All database tables exist!")
        return True
        
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        return False

def start_agent():
    """Start the SQL enhanced agent"""
    print("\n🚀 Starting SQL Enhanced RAG Chatbot...")
    print("=" * 50)
    
    try:
        # Start the agent
        subprocess.run([
            sys.executable, 
            "sql_enhanced_agent.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Agent stopped by user")
    except Exception as e:
        print(f"❌ Error starting agent: {e}")

def main():
    """Main function"""
    print("🚀 SQL Enhanced RAG Chatbot Setup")
    print("=" * 60)
    
    # Check dependencies
    if not check_python_dependencies():
        return
    
    # Check environment variables
    if not check_environment_variables():
        return
    
    # Check database connection
    if not check_database_connection():
        return
    
    # Check database tables
    if not check_database_tables():
        return
    
    print("\n🎉 All checks passed! Starting agent...")
    start_agent()

if __name__ == "__main__":
    main()
