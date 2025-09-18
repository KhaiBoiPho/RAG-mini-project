import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Scale, Car, AlertCircle, Clock, User, Bot, Loader, 
  Home, MessageCircle, HelpCircle, MessageSquare, Menu, X,
  Shield, BookOpen, Users, Award, ChevronRight, Star, 
  Mail, Phone, MapPin, Trash2, RefreshCw
} from 'lucide-react';

// API Service
const API_BASE = 'http://localhost:8000';

const apiService = {
  async sendMessage(messages, conversationId, stream = false) {
    const response = await fetch(`${API_BASE}/chat${stream ? '/stream' : ''}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messages,
        conversation_id: conversationId,
        model: 'gpt-4o-mini',
        stream: stream,
        top_k: 5
      })
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return response;
  },

  async clearConversation(conversationId) {
    const response = await fetch(`${API_BASE}/conversations/${conversationId}`, {
      method: 'DELETE'
    });
    if (!response.ok) throw new Error('Failed to clear conversation');
    return response.json();
  },

  async submitFeedback(feedback) {
    // Mock API call - replace with actual endpoint
    return new Promise((resolve) => {
      setTimeout(() => resolve({ success: true }), 1000);
    });
  }
};

// Layout Components
const Header = ({ activeTab, onTabChange, isMobileMenuOpen, setIsMobileMenuOpen }) => (
  <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50">
    <div className="max-w-7xl mx-auto px-4">
      <div className="flex items-center justify-between h-16">
        <div className="flex items-center space-x-3">
          <div className="bg-gradient-to-r from-emerald-500 to-blue-500 p-2 rounded-xl">
            <Scale className="h-7 w-7 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-emerald-600 to-blue-600 bg-clip-text text-transparent">
              TrafficLaw AI
            </h1>
            <p className="text-xs text-gray-500">Trợ lý pháp luật giao thông</p>
          </div>
        </div>
        
        {/* Desktop Navigation */}
        <nav className="hidden md:flex space-x-8">
          {[
            { id: 'home', label: 'Trang chủ', icon: Home },
            { id: 'chat', label: 'Trò chuyện', icon: MessageCircle },
            { id: 'faq', label: 'FAQ', icon: HelpCircle },
            { id: 'feedback', label: 'Góp ý', icon: MessageSquare }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all ${
                activeTab === id
                  ? 'bg-gradient-to-r from-emerald-100 to-blue-100 text-emerald-700 border border-emerald-200'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
              }`}
            >
              <Icon size={18} />
              <span className="font-medium">{label}</span>
            </button>
          ))}
        </nav>

        {/* Mobile menu button */}
        <button
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="md:hidden p-2 rounded-lg text-gray-600 hover:bg-gray-100"
        >
          {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
    </div>

    {/* Mobile Navigation */}
    {isMobileMenuOpen && (
      <div className="md:hidden bg-white border-t border-gray-200">
        <div className="px-4 py-3 space-y-2">
          {[
            { id: 'home', label: 'Trang chủ', icon: Home },
            { id: 'chat', label: 'Trò chuyện', icon: MessageCircle },
            { id: 'faq', label: 'FAQ', icon: HelpCircle },
            { id: 'feedback', label: 'Góp ý', icon: MessageSquare }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => {
                onTabChange(id);
                setIsMobileMenuOpen(false);
              }}
              className={`w-full flex items-center space-x-3 px-3 py-3 rounded-lg transition-all ${
                activeTab === id
                  ? 'bg-gradient-to-r from-emerald-100 to-blue-100 text-emerald-700'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <Icon size={20} />
              <span className="font-medium">{label}</span>
            </button>
          ))}
        </div>
      </div>
    )}
  </header>
);

// Home Page Component
const HomePage = ({ onNavigateToChat }) => (
  <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-white to-blue-50">
    {/* Hero Section */}
    <section className="py-20 px-4">
      <div className="max-w-6xl mx-auto text-center">
        <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-emerald-100 to-blue-100 px-6 py-2 rounded-full mb-8">
          <Shield className="h-5 w-5 text-emerald-600" />
          <span className="text-emerald-700 font-medium">AI-Powered Legal Assistant</span>
        </div>
        
        <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-emerald-600 via-blue-600 to-purple-600 bg-clip-text text-transparent mb-6">
          Trợ lý pháp luật<br />giao thông thông minh
        </h1>
        
        <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-12 leading-relaxed">
          Tìm hiểu mọi thông tin về luật giao thông đường bộ Việt Nam một cách nhanh chóng, chính xác với công nghệ AI tiên tiến
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={onNavigateToChat}
            className="px-8 py-4 bg-gradient-to-r from-emerald-500 to-blue-500 text-white rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 flex items-center justify-center space-x-2"
          >
            <MessageCircle className="h-6 w-6" />
            <span>Bắt đầu trò chuyện</span>
          </button>
          
          <button className="px-8 py-4 bg-white border-2 border-gray-200 text-gray-700 rounded-xl font-semibold text-lg hover:border-emerald-300 hover:text-emerald-600 transition-all duration-300 flex items-center justify-center space-x-2">
            <BookOpen className="h-6 w-6" />
            <span>Tìm hiểu thêm</span>
          </button>
        </div>
      </div>
    </section>

    {/* Features Section */}
    <section className="py-20 px-4 bg-white">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-800 mb-4">Tính năng nổi bật</h2>
          <p className="text-xl text-gray-600">Những gì làm cho TrafficLaw AI trở nên đặc biệt</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            {
              icon: Bot,
              title: "AI Thông minh",
              description: "Sử dụng công nghệ AI tiên tiến để phân tích và trả lời câu hỏi một cách chính xác nhất",
              gradient: "from-emerald-500 to-teal-500"
            },
            {
              icon: BookOpen,
              title: "Cơ sở dữ liệu đầy đủ",
              description: "Cập nhật liên tục với các văn bản pháp luật mới nhất về giao thông đường bộ",
              gradient: "from-blue-500 to-indigo-500"
            },
            {
              icon: Users,
              title: "Dễ sử dụng",
              description: "Giao diện thân thiện, dễ hiểu, phù hợp cho mọi đối tượng người dùng",
              gradient: "from-purple-500 to-pink-500"
            }
          ].map((feature, idx) => (
            <div key={idx} className="group bg-white p-8 rounded-2xl shadow-lg border border-gray-100 hover:shadow-2xl transform hover:-translate-y-2 transition-all duration-300">
              <div className={`w-16 h-16 bg-gradient-to-r ${feature.gradient} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <feature.icon className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-800 mb-4">{feature.title}</h3>
              <p className="text-gray-600 leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>

    {/* Stats Section */}
    <section className="py-20 px-4 bg-gradient-to-r from-emerald-50 to-blue-50">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {[
            { number: "1000+", label: "Câu hỏi đã trả lời" },
            { number: "99%", label: "Độ chính xác" },
            { number: "24/7", label: "Hỗ trợ liên tục" },
            { number: "5⭐", label: "Đánh giá người dùng" }
          ].map((stat, idx) => (
            <div key={idx} className="text-center">
              <div className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-blue-600 bg-clip-text text-transparent mb-2">
                {stat.number}
              </div>
              <div className="text-gray-600 font-medium">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  </div>
);

// Chat Components
const Message = ({ message, isUser, timestamp, sources }) => {
  const [showSources, setShowSources] = useState(false);

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`flex max-w-4xl ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start`}>
        <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center shadow-md ${
          isUser 
            ? 'bg-gradient-to-r from-blue-500 to-indigo-500 ml-3' 
            : 'bg-gradient-to-r from-emerald-500 to-teal-500 mr-3'
        }`}>
          {isUser ? <User size={18} className="text-white" /> : <Bot size={18} className="text-white" />}
        </div>
        
        <div className={`rounded-2xl px-6 py-4 shadow-lg max-w-2xl ${
          isUser 
            ? 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white' 
            : 'bg-white border border-gray-100 text-gray-800'
        }`}>
          <div className="whitespace-pre-wrap leading-relaxed">{message}</div>
          
          {timestamp && (
            <div className={`text-xs mt-3 flex items-center ${
              isUser ? 'text-blue-100' : 'text-gray-500'
            }`}>
              <Clock size={12} className="mr-1" />
              {new Date(timestamp).toLocaleTimeString('vi-VN')}
            </div>
          )}
          
          {sources && sources.length > 0 && !isUser && (
            <div className="mt-4">
              <button
                onClick={() => setShowSources(!showSources)}
                className="text-sm text-emerald-600 hover:text-emerald-700 flex items-center font-medium"
              >
                <Scale size={14} className="mr-1" />
                Nguồn tham khảo ({sources.length})
                <ChevronRight 
                  size={14} 
                  className={`ml-1 transform transition-transform ${showSources ? 'rotate-90' : ''}`} 
                />
              </button>
              
              {showSources && (
                <div className="mt-3 space-y-3">
                  {sources.slice(0, 3).map((source, idx) => (
                    <div key={idx} className="bg-gradient-to-r from-emerald-50 to-blue-50 p-4 rounded-xl border-l-4 border-emerald-400">
                      <div className="text-sm text-gray-700 leading-relaxed">
                        {source.content.substring(0, 200)}...
                      </div>
                      {source.metadata?.source && (
                        <div className="text-xs text-emerald-600 mt-2 font-medium">
                          📄 {source.metadata.source}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const TypingIndicator = () => (
  <div className="flex justify-start mb-6">
    <div className="flex items-start">
      <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-r from-emerald-500 to-teal-500 mr-3 flex items-center justify-center shadow-md">
        <Bot size={18} className="text-white" />
      </div>
      <div className="bg-white border border-gray-100 rounded-2xl px-6 py-4 shadow-lg">
        <div className="flex items-center space-x-3">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
          <span className="text-sm text-gray-600">Đang tìm kiếm thông tin pháp luật...</span>
        </div>
      </div>
    </div>
  </div>
);

const ChatSidebar = ({ onQuestionSelect, isVisible, onClose }) => {
  const popularQuestions = [
    {
      category: "Vi phạm tốc độ",
      questions: [
        "Mức phạt vi phạm tốc độ ô tô là bao nhiều?",
        "Phạt bao nhiêu khi chạy quá tốc độ cho phép 20km/h?",
        "Quy định tốc độ tối đa trong khu dân cư?"
      ]
    },
    {
      category: "Bằng lái xe",
      questions: [
        "Điều kiện để được cấp bằng lái xe máy?",
        "Thủ tục gia hạn bằng lái xe ô tô?",
        "Mất bằng lái xe phải làm thế nào?"
      ]
    },
    {
      category: "Nồng độ cồn",
      questions: [
        "Quy định về nồng độ cồn khi lái xe?",
        "Mức phạt vi phạm nồng độ cồn 2024?",
        "Cách tính nồng độ cồn trong máu?"
      ]
    },
    {
      category: "An toàn giao thông",
      questions: [
        "Phạt bao nhiêu khi không đội mũ bảo hiểm?",
        "Quy định về đỗ xe trên vỉa hè?",
        "Luật về việc sử dụng điện thoại khi lái xe?"
      ]
    }
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isVisible && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed top-0 left-0 h-full w-80 bg-white shadow-2xl transform transition-transform duration-300 z-50
        lg:relative lg:translate-x-0 lg:shadow-lg
        ${isVisible ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <h3 className="text-lg font-bold text-gray-800">Câu hỏi phổ biến</h3>
          <button
            onClick={onClose}
            className="lg:hidden p-2 rounded-lg hover:bg-gray-100"
          >
            <X size={20} />
          </button>
        </div>
        
        <div className="p-6 overflow-y-auto h-full">
          <div className="space-y-6">
            {popularQuestions.map((category, idx) => (
              <div key={idx} className="space-y-3">
                <h4 className="font-semibold text-emerald-700 flex items-center">
                  <Car size={16} className="mr-2" />
                  {category.category}
                </h4>
                <div className="space-y-2">
                  {category.questions.map((question, qIdx) => (
                    <button
                      key={qIdx}
                      onClick={() => {
                        onQuestionSelect(question);
                        onClose();
                      }}
                      className="w-full text-left p-3 bg-gradient-to-r from-gray-50 to-gray-100 hover:from-emerald-50 hover:to-blue-50 rounded-xl text-sm text-gray-700 hover:text-emerald-700 transition-all duration-200 hover:shadow-md"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
};

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId] = useState(() => `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [isConnected, setIsConnected] = useState(true);
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setMessages([{
      id: 'welcome',
      content: '👋 Xin chào! Tôi là TrafficLaw AI - trợ lý pháp luật giao thông thông minh.\n\nTôi có thể giúp bạn:\n\n🚗 Tìm hiểu mức phạt vi phạm giao thông\n📋 Hướng dẫn thủ tục bằng lái xe\n⚖️ Giải thích các quy định an toàn giao thông\n📱 Tư vấn về luật giao thông mới nhất\n\nHãy đặt câu hỏi hoặc chọn từ danh sách câu hỏi phổ biến bên trái!',
      isUser: false,
      timestamp: new Date(),
      sources: null
    }]);
  }, []);

  const handleSendMessage = async (messageText = inputMessage) => {
    if (!messageText.trim() || isLoading) return;

    const userMessage = {
      id: `user_${Date.now()}`,
      content: messageText.trim(),
      isUser: true,
      timestamp: new Date(),
      sources: null
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const apiMessages = [...messages, userMessage].map(msg => ({
        role: msg.isUser ? 'user' : 'assistant',
        content: msg.content,
        timestamp: msg.timestamp
      }));

      const response = await apiService.sendMessage(apiMessages, conversationId, false);
      const data = await response.json();

      if (data.choices && data.choices[0]) {
        const assistantMessage = {
          id: `assistant_${Date.now()}`,
          content: data.choices[0].message.content,
          isUser: false,
          timestamp: new Date(),
          sources: data.sources || null
        };

        setMessages(prev => [...prev, assistantMessage]);
      }

      setIsConnected(true);
    } catch (error) {
      console.error('Error sending message:', error);
      setIsConnected(false);
      
      const errorMessage = {
        id: `error_${Date.now()}`,
        content: '😔 Xin lỗi, đã xảy ra lỗi khi kết nối với server. Vui lòng thử lại sau hoặc kiểm tra kết nối mạng.',
        isUser: false,
        timestamp: new Date(),
        sources: null
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearConversation = async () => {
    try {
      await apiService.clearConversation(conversationId);
      setMessages([{
        id: 'welcome_new',
        content: '🔄 Cuộc trò chuyện đã được làm mới. Bạn có câu hỏi gì về pháp luật giao thông không?',
        isUser: false,
        timestamp: new Date(),
        sources: null
      }]);
    } catch (error) {
      console.error('Error clearing conversation:', error);
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-emerald-50 via-white to-blue-50">
      <ChatSidebar 
        onQuestionSelect={handleSendMessage}
        isVisible={isSidebarVisible}
        onClose={() => setIsSidebarVisible(false)}
      />
      
      <div className="flex-1 flex flex-col">
        {/* Chat Header */}
        <div className="bg-white border-b border-gray-200 shadow-sm p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setIsSidebarVisible(true)}
                className="lg:hidden p-2 rounded-lg hover:bg-gray-100"
              >
                <Menu size={20} />
              </button>
              <div className="flex items-center space-x-3">
                <div className="bg-gradient-to-r from-emerald-500 to-teal-500 p-2 rounded-lg">
                  <MessageCircle className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="font-bold text-gray-800">Trò chuyện với AI</h2>
                  <div className="flex items-center space-x-2 text-sm">
                    <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
                    <span className="text-gray-600">
                      {isConnected ? 'Đã kết nối' : 'Mất kết nối'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={handleClearConversation}
                className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                title="Làm mới cuộc trò chuyện"
              >
                <RefreshCw size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-4xl mx-auto">
            {messages.map((message) => (
              <Message
                key={message.id}
                message={message.content}
                isUser={message.isUser}
                timestamp={message.timestamp}
                sources={message.sources}
              />
            ))}
            
            {isLoading && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-6">
          <div className="max-w-4xl mx-auto">
            <div className="flex space-x-4">
              <div className="flex-1">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder="Nhập câu hỏi về pháp luật giao thông..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none bg-gray-50 focus:bg-white transition-all"
                  rows="1"
                  disabled={isLoading}
                />
              </div>
              <button
                onClick={() => handleSendMessage()}
                disabled={!inputMessage.trim() || isLoading}
                className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-blue-500 text-white rounded-xl font-semibold hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 transition-all duration-300 hover:-translate-y-0.5"
              >
                {isLoading ? (
                  <Loader className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
                <span>Gửi</span>
              </button>
            </div>
            
            <div className="mt-4 flex items-center justify-center space-x-6 text-xs text-gray-500">
              <div className="flex items-center space-x-1">
                <Shield size={12} />
                <span>Thông tin được mã hóa bảo mật</span>
              </div>
              <div className="flex items-center space-x-1">
                <AlertCircle size={12} />
                <span>Thông tin mang tính chất tham khảo</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// FAQ Page Component
const FAQPage = () => {
  const [openFAQ, setOpenFAQ] = useState(null);
  
  const faqs = [
    {
      category: "Về TrafficLaw AI",
      items: [
        {
          question: "TrafficLaw AI là gì?",
          answer: "TrafficLaw AI là trợ lý thông minh sử dụng công nghệ trí tuệ nhân tạo để cung cấp thông tin chính xác về pháp luật giao thông đường bộ Việt Nam. Hệ thống được xây dựng dựa trên cơ sở dữ liệu pháp luật cập nhật và công nghệ xử lý ngôn ngữ tự nhiên tiên tiến."
        },
        {
          question: "Thông tin từ TrafficLaw AI có chính xác không?",
          answer: "Chúng tôi luôn cập nhật cơ sở dữ liệu với các văn bản pháp luật mới nhất. Tuy nhiên, thông tin chỉ mang tính chất tham khảo. Đối với các vấn đề pháp lý phức tạp, bạn nên tham khảo ý kiến của luật sư hoặc cơ quan có thẩm quyền."
        },
        {
          question: "TrafficLaw AI có miễn phí không?",
          answer: "Hiện tại, TrafficLaw AI hoàn toàn miễn phí cho tất cả người dùng. Chúng tôi cam kết cung cấp dịch vụ chất lượng cao để hỗ trợ cộng đồng tìm hiểu pháp luật giao thông."
        }
      ]
    },
    {
      category: "Vi phạm giao thông",
      items: [
        {
          question: "Làm thế nào để biết mức phạt vi phạm giao thông?",
          answer: "Bạn có thể hỏi trực tiếp với TrafficLaw AI về loại vi phạm cụ thể. Ví dụ: 'Phạt bao nhiêu khi chạy quá tốc độ 20km/h?' hoặc 'Mức phạt khi không đội mũ bảo hiểm?'. Hệ thống sẽ cung cấp thông tin chi tiết về mức phạt theo quy định hiện hành."
        },
        {
          question: "Có thể tra cứu quy định về đỗ xe không?",
          answer: "Có thể! TrafficLaw AI có thể giải đáp các câu hỏi về quy định đỗ xe như: nơi được phép đỗ xe, nơi cấm đỗ xe, mức phạt vi phạm quy định đỗ xe, và các trường hợp đặc biệt khác."
        },
        {
          question: "Làm thế nào để biết quy định tốc độ?",
          answer: "Bạn có thể hỏi về quy định tốc độ cho từng loại đường, từng loại phương tiện. Ví dụ: 'Tốc độ tối đa trong khu dân cư?', 'Tốc độ cho phép trên cao tốc?', hoặc 'Quy định tốc độ cho xe máy?'."
        }
      ]
    },
    {
      category: "Bằng lái xe",
      items: [
        {
          question: "Có thể tra cứu thông tin về bằng lái xe không?",
          answer: "Hoàn toàn có thể! TrafficLaw AI cung cấp thông tin về: điều kiện cấp bằng lái, thủ tục đăng ký thi bằng lái, quy trình gia hạn bằng lái, xử lý khi mất bằng lái, và các quy định về sử dụng bằng lái xe."
        },
        {
          question: "Thủ tục thi bằng lái xe như thế nào?",
          answer: "Bạn có thể hỏi chi tiết về: hồ sơ cần thiết, điều kiện sức khỏe, quy trình đăng ký thi, nội dung thi lý thuyết và thực hành, lệ phí thi bằng lái cho từng hạng xe cụ thể."
        }
      ]
    },
    {
      category: "Sử dụng hệ thống",
      items: [
        {
          question: "Làm thế nào để đặt câu hỏi hiệu quả?",
          answer: "Để nhận được câu trả lời chính xác nhất, bạn nên: (1) Đặt câu hỏi cụ thể và rõ ràng, (2) Nêu rõ loại phương tiện (ô tô, xe máy, xe tải...), (3) Đề cập đến tình huống cụ thể nếu có, (4) Sử dụng các từ khóa liên quan đến pháp luật giao thông."
        },
        {
          question: "Có thể hỏi nhiều câu hỏi trong một cuộc trò chuyện không?",
          answer: "Có thể! Hệ thống lưu trữ lịch sử cuộc trò chuyện và có thể tham chiếu đến các câu hỏi trước đó. Bạn có thể đặt câu hỏi liên tục hoặc yêu cầu làm rõ thêm thông tin."
        },
        {
          question: "Dữ liệu cá nhân có được bảo mật không?",
          answer: "Chúng tôi cam kết bảo vệ quyền riêng tư của người dùng. Các cuộc trò chuyện được mã hóa và không lưu trữ thông tin cá nhân nhận dạng. Dữ liệu chỉ được sử dụng để cải thiện chất lượng dịch vụ."
        }
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-white to-blue-50 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-emerald-100 to-blue-100 px-6 py-2 rounded-full mb-6">
            <HelpCircle className="h-5 w-5 text-emerald-600" />
            <span className="text-emerald-700 font-medium">Câu hỏi thường gặp</span>
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-blue-600 bg-clip-text text-transparent mb-4">
            FAQ
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Tìm câu trả lời nhanh chóng cho những thắc mắc phổ biến về TrafficLaw AI và pháp luật giao thông
          </p>
        </div>

        <div className="space-y-8">
          {faqs.map((category, categoryIdx) => (
            <div key={categoryIdx} className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
              <div className="bg-gradient-to-r from-emerald-500 to-blue-500 px-6 py-4">
                <h2 className="text-xl font-bold text-white flex items-center">
                  <BookOpen className="h-6 w-6 mr-3" />
                  {category.category}
                </h2>
              </div>
              
              <div className="p-6">
                <div className="space-y-4">
                  {category.items.map((item, itemIdx) => {
                    const faqId = `${categoryIdx}-${itemIdx}`;
                    const isOpen = openFAQ === faqId;
                    
                    return (
                      <div key={itemIdx} className="border border-gray-200 rounded-xl overflow-hidden">
                        <button
                          onClick={() => setOpenFAQ(isOpen ? null : faqId)}
                          className="w-full px-6 py-4 text-left hover:bg-gray-50 transition-colors flex items-center justify-between"
                        >
                          <span className="font-semibold text-gray-800 pr-4">{item.question}</span>
                          <ChevronRight 
                            className={`h-5 w-5 text-gray-500 transform transition-transform ${
                              isOpen ? 'rotate-90' : ''
                            }`} 
                          />
                        </button>
                        
                        {isOpen && (
                          <div className="px-6 pb-4 bg-gradient-to-r from-gray-50 to-blue-50">
                            <p className="text-gray-700 leading-relaxed">{item.answer}</p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Contact section */}
        <div className="mt-12 bg-white rounded-2xl shadow-lg border border-gray-100 p-8 text-center">
          <h3 className="text-2xl font-bold text-gray-800 mb-4">Không tìm thấy câu trả lời?</h3>
          <p className="text-gray-600 mb-6">
            Hãy thử đặt câu hỏi trực tiếp với TrafficLaw AI hoặc gửi phản hồi cho chúng tôi
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button 
              onClick={() => window.location.hash = '#chat'}
              className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-blue-500 text-white rounded-xl font-semibold hover:shadow-lg transition-all duration-300"
            >
              <MessageCircle className="h-5 w-5 inline mr-2" />
              Trò chuyện ngay
            </button>
            <button 
              onClick={() => window.location.hash = '#feedback'}
              className="px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-xl font-semibold hover:border-emerald-500 hover:text-emerald-600 transition-all duration-300"
            >
              <MessageSquare className="h-5 w-5 inline mr-2" />
              Gửi phản hồi
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Feedback Page Component
const FeedbackPage = () => {
  const [feedbackData, setFeedbackData] = useState({
    type: 'general',
    subject: '',
    message: '',
    email: '',
    name: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const feedbackTypes = [
    { value: 'general', label: 'Góp ý chung', icon: MessageSquare },
    { value: 'bug', label: 'Báo lỗi', icon: AlertCircle },
    { value: 'feature', label: 'Đề xuất tính năng', icon: Star },
    { value: 'accuracy', label: 'Thông tin không chính xác', icon: Shield }
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      await apiService.submitFeedback(feedbackData);
      setSubmitted(true);
      setFeedbackData({ type: 'general', subject: '', message: '', email: '', name: '' });
    } catch (error) {
      console.error('Error submitting feedback:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-white to-blue-50 flex items-center justify-center px-4">
        <div className="max-w-md mx-auto bg-white rounded-2xl shadow-lg border border-gray-100 p-8 text-center">
          <div className="w-16 h-16 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-6">
            <Star className="h-8 w-8 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Cảm ơn bạn!</h2>
          <p className="text-gray-600 mb-6">
            Phản hồi của bạn đã được gửi thành công. Chúng tôi sẽ xem xét và cải thiện dịch vụ dựa trên góp ý của bạn.
          </p>
          <button
            onClick={() => setSubmitted(false)}
            className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-blue-500 text-white rounded-xl font-semibold hover:shadow-lg transition-all duration-300"
          >
            Gửi phản hồi khác
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-white to-blue-50 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-emerald-100 to-blue-100 px-6 py-2 rounded-full mb-6">
            <MessageSquare className="h-5 w-5 text-emerald-600" />
            <span className="text-emerald-700 font-medium">Phản hồi & Góp ý</span>
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-blue-600 bg-clip-text text-transparent mb-4">
            Liên hệ với chúng tôi
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Ý kiến của bạn rất quan trọng để chúng tôi cải thiện TrafficLaw AI. Hãy chia sẻ trải nghiệm và đề xuất của bạn!
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Contact Info */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6 mb-6">
              <h3 className="text-xl font-bold text-gray-800 mb-6">Thông tin liên hệ</h3>
              
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg flex items-center justify-center">
                    <Mail className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800">Email</p>
                    <p className="text-gray-600 text-sm">support@trafficlaw.ai</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
                    <Phone className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800">Hotline</p>
                    <p className="text-gray-600 text-sm">1900 1234 (8:00 - 22:00)</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                    <MapPin className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800">Địa chỉ</p>
                    <p className="text-gray-600 text-sm">TP. Hồ Chí Minh, Việt Nam</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-emerald-500 to-blue-500 rounded-2xl p-6 text-white">
              <h3 className="text-xl font-bold mb-4">Cam kết của chúng tôi</h3>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center">
                  <Star className="h-4 w-4 mr-2" />
                  Phản hồi trong 24h
                </li>
                <li className="flex items-center">
                  <Shield className="h-4 w-4 mr-2" />
                  Bảo mật thông tin
                </li>
                <li className="flex items-center">
                  <Award className="h-4 w-4 mr-2" />
                  Cải thiện liên tục
                </li>
              </ul>
            </div>
          </div>

          {/* Feedback Form */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Feedback Type */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-3">
                    Loại phản hồi *
                  </label>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {feedbackTypes.map((type) => (
                      <button
                        key={type.value}
                        type="button"
                        onClick={() => setFeedbackData({ ...feedbackData, type: type.value })}
                        className={`p-4 rounded-xl border-2 transition-all text-left ${
                          feedbackData.type === type.value
                            ? 'border-emerald-500 bg-emerald-50 text-emerald-700'
                            : 'border-gray-200 hover:border-gray-300 text-gray-700'
                        }`}
                      >
                        <type.icon className="h-5 w-5 mb-2" />
                        <div className="font-semibold">{type.label}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Personal Info */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Họ tên
                    </label>
                    <input
                      type="text"
                      value={feedbackData.name}
                      onChange={(e) => setFeedbackData({ ...feedbackData, name: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      placeholder="Nguyễn Văn A"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Email *
                    </label>
                    <input
                      type="email"
                      required
                      value={feedbackData.email}
                      onChange={(e) => setFeedbackData({ ...feedbackData, email: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      placeholder="example@email.com"
                    />
                  </div>
                </div>

                {/* Subject */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Tiêu đề *
                  </label>
                  <input
                    type="text"
                    required
                    value={feedbackData.subject}
                    onChange={(e) => setFeedbackData({ ...feedbackData, subject: e.target.value })}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    placeholder="Tóm tắt nội dung phản hồi"
                  />
                </div>

                {/* Message */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Nội dung chi tiết *
                  </label>
                  <textarea
                    required
                    rows="6"
                    value={feedbackData.message}
                    onChange={(e) => setFeedbackData({ ...feedbackData, message: e.target.value })}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
                    placeholder="Mô tả chi tiết phản hồi của bạn..."
                  />
                </div>

                {/* Submit Button */}
                <div className="flex justify-end">
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="px-8 py-3 bg-gradient-to-r from-emerald-500 to-blue-500 text-white rounded-xl font-semibold hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 transition-all duration-300"
                  >
                    {isSubmitting ? (
                      <Loader className="w-5 h-5 animate-spin" />
                    ) : (
                      <Send className="w-5 h-5" />
                    )}
                    <span>{isSubmitting ? 'Đang gửi...' : 'Gửi phản hồi'}</span>
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main App Component
const VietnamTrafficLawApp = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Handle URL hash navigation
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1);
      if (['home', 'chat', 'faq', 'feedback'].includes(hash)) {
        setActiveTab(hash);
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    handleHashChange(); // Check initial hash
    
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    window.location.hash = tab;
    setIsMobileMenuOpen(false);
  };

  const renderCurrentPage = () => {
    switch (activeTab) {
      case 'chat':
        return <ChatPage />;
      case 'faq':
        return <FAQPage />;
      case 'feedback':
        return <FeedbackPage />;
      default:
        return <HomePage onNavigateToChat={() => handleTabChange('chat')} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {activeTab !== 'chat' && (
        <Header 
          activeTab={activeTab}
          onTabChange={handleTabChange}
          isMobileMenuOpen={isMobileMenuOpen}
          setIsMobileMenuOpen={setIsMobileMenuOpen}
        />
      )}
      {renderCurrentPage()}
    </div>
  );
};

export default VietnamTrafficLawApp;