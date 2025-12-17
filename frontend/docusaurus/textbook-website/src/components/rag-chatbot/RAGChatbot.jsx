import React, { useState, useEffect, useRef } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { defaultRAGService } from './rag-api';
import './rag-chatbot.css';

const RAGChatbot = ({ title = "Physical AI & Robotics Assistant" }) => {
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your Physical AI & Humanoid Robotics assistant. How can I help you with the textbook content today?", sender: 'bot' }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(true);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Check connection status on mount
  useEffect(() => {
    const checkConnection = async () => {
      const connected = await defaultRAGService.healthCheck();
      setIsConnected(connected);

      // If connection fails, add a message to inform the user
      if (!connected) {
        setMessages(prev => [
          ...prev,
          {
            id: Date.now(),
            text: "⚠️ Note: The RAG backend service is currently unavailable. The chatbot is running in simulation mode.",
            sender: 'system'
          }
        ]);
      }
    };

    checkConnection();
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputText,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      if (isConnected) {
        // Call the real RAG API
        const result = await defaultRAGService.query(inputText);

        if (result.success) {
          const botResponse = {
            id: Date.now() + 1,
            text: result.response,
            sender: 'bot',
            sources: result.sources
          };
          setMessages(prev => [...prev, botResponse]);
        } else {
          const errorResponse = {
            id: Date.now() + 1,
            text: `Sorry, I encountered an error processing your request: ${result.error}. Please try again.`,
            sender: 'bot'
          };
          setMessages(prev => [...prev, errorResponse]);
        }
      } else {
        // Fallback to simulated response if backend is not available
        await new Promise(resolve => setTimeout(resolve, 1000));

        const simulatedResponse = {
          id: Date.now() + 1,
          text: `I received your question: "${inputText}". In a real implementation with the RAG backend connected, this would query the textbook content and provide an accurate answer based on the Physical AI & Humanoid Robotics textbook.`,
          sender: 'bot'
        };

        setMessages(prev => [...prev, simulatedResponse]);
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I encountered an error processing your request. Please try again.",
        sender: 'bot'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <BrowserOnly>
      {() => (
        <div className="rag-chatbot-container">
          <div className="rag-chatbot-header">
            <h3>{title}</h3>
            <div className="chatbot-status">
              <span className={`status-indicator ${isConnected ? (isLoading ? 'status-thinking' : 'status-online') : 'status-offline'}`}></span>
              <span className="status-text">
                {isLoading ? 'Thinking...' : isConnected ? 'Online' : 'Backend Disconnected'}
              </span>
            </div>
          </div>

          <div className="rag-chatbot-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.sender}-message`}
              >
                <div className="message-content">
                  {message.text}
                  {message.sources && message.sources.length > 0 && (
                    <details className="message-sources">
                      <summary>Sources</summary>
                      <ul>
                        {message.sources.map((source, idx) => (
                          <li key={idx}>{source.title || source.text.substring(0, 50) + '...'}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
                <div className="message-sender">
                  {message.sender === 'bot' ? '🤖 Assistant' :
                   message.sender === 'system' ? '🔧 System' : '👤 You'}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot-message">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
                <div className="message-sender">🤖 Assistant</div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form className="rag-chatbot-input-form" onSubmit={handleSendMessage}>
            <input
              type="text"
              ref={inputRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Ask a question about Physical AI & Robotics..."
              disabled={isLoading}
              className="chatbot-input"
            />
            <button
              type="submit"
              disabled={!inputText.trim() || isLoading}
              className="chatbot-send-button"
            >
              Send
            </button>
          </form>
        </div>
      )}
    </BrowserOnly>
  );
};

export default RAGChatbot;