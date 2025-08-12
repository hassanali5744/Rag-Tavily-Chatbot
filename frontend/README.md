# 🤖 Enhanced RAG Chatbot - Frontend

A modern React-based frontend for the Enhanced RAG Chatbot that provides an intuitive chat interface with document management capabilities.

## ✨ Features

- **💬 Real-time Chat Interface**: Beautiful, responsive chat UI with loading states
- **📚 Document Management**: Upload, view, and delete documents from the knowledge base
- **🔍 Source Attribution**: See which documents and web sources were used in responses
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **⚡ Fast Performance**: Built with Vite for lightning-fast development and builds
- **🎨 Modern UI**: Clean, intuitive interface with smooth animations

## 🏗️ Architecture

The frontend is built with:
- **React 19** - Latest React with modern features
- **Vite** - Fast build tool and development server
- **CSS3** - Custom styling with modern design patterns
- **Fetch API** - For communication with the Python backend

## 🚀 Quick Start

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn package manager
- The Python backend must be running (see main README)

### Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Navigate to `http://localhost:5173`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## 📁 Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── App.jsx            # Main React component
│   ├── App.css            # Main stylesheet
│   └── main.jsx           # Application entry point
├── package.json           # Dependencies and scripts
├── vite.config.js         # Vite configuration
├── eslint.config.js       # ESLint configuration
└── index.html             # HTML template
```

## 🎯 Key Components

### App.jsx
The main React component that handles:
- Chat interface and message display
- Document upload and management
- API communication with the backend
- State management for the application

### App.css
Comprehensive styling including:
- Modern chat interface design
- Responsive layout for all screen sizes
- Loading animations and transitions
- Document management UI

## 🔧 Configuration

### Environment Variables
The frontend connects to the Python backend by default at `http://localhost:8000`. If you need to change this:

1. Create a `.env` file in the frontend directory
2. Add: `VITE_API_URL=http://your-backend-url:port`

### API Endpoints Used

The frontend communicates with these backend endpoints:
- `POST /chat` - Send chat messages
- `POST /upload_document` - Upload new documents
- `GET /documents` - List all documents
- `DELETE /documents/{title}` - Delete documents
- `GET /health` - Health check

## 🎨 UI Features

### Chat Interface
- **Message Bubbles**: User and AI messages with distinct styling
- **Loading States**: Animated loading indicators during API calls
- **Source Links**: Clickable links to document sources
- **Auto-scroll**: Automatically scrolls to new messages

### Document Management
- **Upload Form**: Simple form for adding new documents
- **Document List**: Sidebar showing all uploaded documents
- **Delete Functionality**: Remove documents with confirmation
- **Chunk Count**: Shows how many chunks each document was split into

### Responsive Design
- **Mobile-first**: Optimized for mobile devices
- **Tablet Support**: Adapts to tablet screen sizes
- **Desktop Enhancement**: Enhanced features on larger screens

## 🛠️ Development

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
```

### Development Tips

1. **Hot Reload**: Changes are automatically reflected in the browser
2. **ESLint**: Code quality is enforced with ESLint
3. **Fast Refresh**: React components update without losing state
4. **Error Overlay**: Clear error messages during development

## 🔍 Troubleshooting

### Common Issues

1. **"Failed to fetch" errors**
   - Ensure the Python backend is running on port 8000
   - Check if the backend is accessible at `http://localhost:8000/health`

2. **CORS errors**
   - The backend should have CORS configured for `http://localhost:5173`
   - Check the backend CORS settings

3. **Build errors**
   - Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
   - Check for syntax errors in your code

4. **Port conflicts**
   - If port 5173 is in use, Vite will automatically try the next available port
   - Check the terminal output for the actual URL

### Debug Mode

Enable debug logging by opening browser DevTools and checking the Console tab for detailed error messages.

## 🚀 Deployment

### Static Hosting
The frontend can be deployed to any static hosting service:

1. **Build the project:**
   ```bash
   npm run build
   ```

2. **Deploy the `dist/` folder** to your hosting service

### Popular Hosting Options
- **Vercel**: `vercel --prod`
- **Netlify**: Drag and drop the `dist/` folder
- **GitHub Pages**: Use GitHub Actions for automatic deployment
- **AWS S3**: Upload to S3 bucket with CloudFront

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Happy coding! 🚀✨**
