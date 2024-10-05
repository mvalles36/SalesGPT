import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid'; // For generating unique session_id
import { Input } from "@/components/ui/input";
import BotIcon from '@/components/ui/bot-icon';
import LoaderIcon from '@/components/ui/loader-icon';
import styles from './ChatInterface.module.css';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import { PostHog } from 'posthog-node';

let client: PostHog | undefined;

const getHeaders = () => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (process.env.NEXT_PUBLIC_ENVIRONMENT === "production") {
    headers['Authorization'] = `Bearer ${process.env.NEXT_PUBLIC_AUTH_KEY}`;
  }
  return headers;
};

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [session_id] = useState(uuidv4()); // Unique session_id generated when the component mounts
  const [stream, setStream] = useState(false);
  const [botName, setBotName] = useState('');
  const [conversationalStage, setConversationalStage] = useState('');
  const [thinkingProcess, setThinkingProcess] = useState<{
    conversationalStage: string,
    tool?: string,
    toolInput?: string,
    actionOutput?: string,
    actionInput?: string
  }[]>([]);
  const [maxHeight, setMaxHeight] = useState('80vh'); // Default to 100% of the viewport height
  const [isBotTyping, setIsBotTyping] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const thinkingProcessEndRef = useRef<null | HTMLDivElement>(null);
  const [botHasResponded, setBotHasResponded] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined' && process.env.NEXT_PUBLIC_ENVIRONMENT === "production") {
      client = new PostHog(`${process.env.NEXT_PUBLIC_POSTHOG_ID}`, {
        host: 'https://app.posthog.com',
        disableGeoip: false,
        requestTimeout: 30000,
      });
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    thinkingProcessEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [thinkingProcess]);

  useEffect(() => {
    if (botHasResponded) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      thinkingProcessEndRef.current?.scrollIntoView({ behavior: "smooth" });
      setBotHasResponded(false); // Reset the flag
    }
  }, [botHasResponded]);

  useEffect(() => {
    // This function will be called on resize events
    const handleResize = () => {
      setMaxHeight(`${window.innerHeight - 200}px`);
    };

    // Set the initial value when the component mounts
    handleResize();

    // Add the event listener for future resize events
    window.addEventListener('resize', handleResize);

    // Return a cleanup function to remove the event listener when the component unmounts
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const fetchBotName = async () => {
      if (process.env.NEXT_PUBLIC_ENVIRONMENT === "production" && client) {
        client.capture({
          distinctId: session_id,
          event: 'fetched-bot-name',
          properties: {
            $current_url: window.location.href,
          },
        });
      }

      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/botname`, {
          headers: getHeaders(),
        });

        if (!response.ok) {
          throw new Error(`Network response was not ok: ${response.statusText}`);
        }

        const data = await response.json();
        setBotName(data.name); // Save the bot name in the state
        console.log(botName);
      } catch (error) {
        console.error("Failed to fetch the bot's name:", error);
      }
    };

    // Call the function to fetch the bot name
    fetchBotName();
  }, [session_id]); // Only trigger on session_id

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const sendMessage = () => {
    if (!inputValue.trim()) return;

    const userMessage = `${inputValue}`;
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: uuidv4(), text: userMessage, sender: 'user' }
    ]);

    handleBotResponse(inputValue);
    setInputValue('');
  };

  useEffect(() => {
    if (process.env.NODE_ENV === "development") {
      console.log('NEXT_PUBLIC_AUTH_KEY:', process.env.NEXT_PUBLIC_AUTH_KEY);
      console.log('NEXT_PUBLIC_ENVIRONMENT:', process.env.NEXT_PUBLIC_ENVIRONMENT);
      console.log('NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL);
    }
  }, []);

  const handleBotResponse = async (userMessage: string) => {
    if (process.env.NEXT_PUBLIC_ENVIRONMENT === "production" && client) {
      client.capture({
        distinctId: session_id,
        event: 'sent-message',
        properties: {
          $current_url: window.location.href,
        },
      });
    }

    const requestData = {
      session_id,
      human_say: userMessage,
      stream,
    };
    setIsBotTyping(true); // Start showing the typing indicator

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Bot response:', data);

      setBotName(data.bot_name); // Update bot name based on response
      setConversationalStage(data.conversational_stage);
      // Update the thinkingProcess state with new fields from the response
      setThinkingProcess((prevProcess) => [
        ...prevProcess,
        {
          conversationalStage: data.conversational_stage,
          tool: data.tool,
          toolInput: data.tool_input,
          actionOutput: data.action_output,
          actionInput: data.action_input
        }
      ]);

      const botMessage: Message = { id: uuidv4(), text: data.response, sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, botMessage]);

    } catch (error) {
      console.error("Failed to fetch bot's response:", error);
    } finally {
      setIsBotTyping(false); // Stop showing the typing indicator
      setBotHasResponded(true);
    }
  };

  return (
    <div key="1" className="flex flex-col " style={{ height: '89vh' }}>
      <header className="flex items-center justify-center h-16 bg-gray-900 text-white">
        <BotIcon className="animate-wave h-7 w-6 mr-2" />
        <h1 className="text-2xl font-bold">SalesGPT</h1>
      </header>
      <main className="flex flex-row justify-center items-start bg-gray-100 dark:bg-gray-900 p-4" >
        <div className="flex flex-col w-1/2 h-full bg-white rounded-lg shadow-md p-4 mr-4 chat-messages" style={{ maxHeight }}>
          <div className="flex items-center mb-4">
            <BotIcon className="h-6 w-6 text-gray-500 mr-2" />
            <h2 className="text-lg font-semibold">Chat Interface With The Customer</h2>
          </div>
          <div className={`flex-1 overflow-y-auto ${styles.hideScrollbar}`}>
            {messages.map((message) => (
              <div key={message.id} className="flex items-center p-2">
                {message.sender === 'user' ? (
                  <>
                    <span role="img" aria-label="User" className="mr-2">👤</span>
                    <span className={`text-frame p-2 rounded-lg bg-blue-100 dark:bg-blue-900 text-blue-900`}>
                      {message.text}
                    </span>
                  </>
                ) : (
                  <div className="flex w-full justify-between">
                    <div className="flex items-center">
                      <img
                        alt="Bot"
                        className="rounded-full mr-2"
                        src="/maskot.png"
                        style={{ width: 24, height: 24, objectFit: "cover" }}
                      />
                      <span className={`text-frame p-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-900`}>
                        <ReactMarkdown rehypePlugins={[rehypeRaw]} components={{
                          a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-700" />
                        }}>
                          {message.text}
                        </ReactMarkdown>
                      </span>
                    </div>
                 
