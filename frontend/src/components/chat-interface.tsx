import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Input } from "@/components/ui/input";
import BotIcon from '@/components/ui/bot-icon';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import { PostHog } from 'posthog-node';
import styles from './ChatInterface.module.css';

let client: PostHog | undefined;

const getHeaders = () => ({
  'Content-Type': 'application/json',
  ...(process.env.NEXT_PUBLIC_ENVIRONMENT === "production" && {
    'Authorization': `Bearer ${process.env.NEXT_PUBLIC_AUTH_KEY}`,
  }),
});

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [session_id] = useState(uuidv4());
  const [botName, setBotName] = useState('');
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  useEffect(() => {
    // Initialize PostHog
    if (typeof window !== 'undefined' && process.env.NEXT_PUBLIC_ENVIRONMENT === "production") {
      client = new PostHog(`${process.env.NEXT_PUBLIC_POSTHOG_ID}`, {
        host: 'https://app.posthog.com',
      });
    }

    // Fetch the bot name
    const fetchBotName = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/botname`, {
          headers: getHeaders(),
        });
        if (!response.ok) throw new Error(response.statusText);
        const data = await response.json();
        setBotName(data.name);
      } catch (error) {
        console.error("Failed to fetch bot's name:", error);
      }
    };

    fetchBotName();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const sendMessage = () => {
    if (!inputValue.trim()) return;
    const userMessage = { id: uuidv4(), text: inputValue, sender: 'user' };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    handleBotResponse(inputValue);
    setInputValue('');
  };

  const handleBotResponse = async (userMessage: string) => {
    const requestData = { session_id, human_say: userMessage };
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify(requestData),
      });
      const data = await response.json();
      const botMessage = { id: uuidv4(), text: data.response, sender: 'bot' };
      setMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (error) {
      console.error("Failed to fetch bot's response:", error);
    }
  };

  return (
    <div className="flex flex-col" style={{ height: '89vh' }}>
      <header className="flex items-center justify-center h-16 bg-gray-900 text-white">
        <BotIcon className="animate-wave h-7 w-6 mr-2" />
        <h1 className="text-2xl font-bold">{botName || "Loading bot name..."}</h1>
      </header>
      <main className="flex flex-row justify-center items-start bg-gray-100 p-4">
        <div className="flex flex-col w-1/2 h-full bg-white rounded-lg shadow-md p-4">
          <div className="flex-1 overflow-y-auto">
            {messages.map(message => (
              <div key={message.id} className="flex items-center p-2">
                {message.sender === 'user' ? (
                  <span className="text-frame p-2 rounded-lg bg-blue-100">{message.text}</span>
                ) : (
                  <span className="text-frame p-2 rounded-lg bg-gray-200">
                    <ReactMarkdown rehypePlugins={[rehypeRaw]}>
                      {message.text}
                    </ReactMarkdown>
                  </span>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <Input value={inputValue} onChange={handleInputChange} onKeyPress={(e) => e.key === 'Enter' && sendMessage()} />
        </div>
      </main>
    </div>
  );
}
