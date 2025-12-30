import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import ChatInterface from './components/ChatInterface';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import { healthApi } from './lib/api';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.check,
    refetchInterval: 30000,
  });

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header 
        health={health} 
        onMenuClick={() => setSidebarOpen(!sidebarOpen)}
      />
      
      <div className="flex h-[calc(100vh-4rem)]">
        <Sidebar 
          isOpen={sidebarOpen} 
          onClose={() => setSidebarOpen(false)} 
        />
        
        <main className="flex-1 overflow-hidden">
          <ChatInterface />
        </main>
      </div>
    </div>
  );
}

export default App;
