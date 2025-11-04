import { useState, useEffect, useRef } from 'react';
import { OpenAIService } from './services/openai';
import { Message } from './types';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { LoadingIndicator } from './components/LoadingIndicator';
import { ApiKeyConfig } from './components/ApiKeyConfig';
import { AlertCircle, Stethoscope } from 'lucide-react';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openAIService, setOpenAIService] = useState<OpenAIService | null>(
    null
  );
  const [isConfigured, setIsConfigured] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
    const assistantId = import.meta.env.VITE_ASSISTANT_ID;

    if (apiKey && assistantId && apiKey !== 'your_api_key_here') {
      setOpenAIService(new OpenAIService(apiKey, assistantId));
      setIsConfigured(true);
    } else {
      const storedApiKey = localStorage.getItem('openai_api_key');
      const storedAssistantId = localStorage.getItem('openai_assistant_id') || 'asst_zvFuLn2wRlDVEh5SFO2jYsey';

      if (storedApiKey) {
        setOpenAIService(new OpenAIService(storedApiKey, storedAssistantId));
        setIsConfigured(true);
      } else {
        setError(
          'OpenAI API key not configured. Please configure it using the button above.'
        );
      }
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleApiKeySet = (apiKey: string, assistantId: string) => {
    localStorage.setItem('openai_api_key', apiKey);
    localStorage.setItem('openai_assistant_id', assistantId);
    setOpenAIService(new OpenAIService(apiKey, assistantId));
    setIsConfigured(true);
    setError(null);
  };

  const handleSendMessage = async (content: string) => {
    if (!openAIService) {
      setError('OpenAI service not initialized');
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await openAIService.askQuestion(content);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'An unexpected error occurred'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-5">
          <div className="flex items-center gap-3 justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-emerald-600 flex items-center justify-center">
                <Stethoscope className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-semibold text-slate-800">
                  EVP Clinical Assistant
                </h1>
                <p className="text-sm text-slate-600">
                  Urologist-facing support for Enfortumab Vedotin + Pembrolizumab
                  therapy
                </p>
              </div>
            </div>
            <ApiKeyConfig onApiKeySet={handleApiKeySet} isConfigured={isConfigured} />
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-8">
          {messages.length === 0 && !error && (
            <div className="text-center py-16">
              <div className="w-16 h-16 rounded-full bg-emerald-100 flex items-center justify-center mx-auto mb-4">
                <Stethoscope className="w-8 h-8 text-emerald-600" />
              </div>
              <h2 className="text-xl font-semibold text-slate-800 mb-2">
                Welcome to EVP Clinical Assistant
              </h2>
              <p className="text-slate-600 max-w-lg mx-auto">
                This tool provides evidence-based clinical guidance for
                urologists managing patients receiving Enfortumab Vedotin +
                Pembrolizumab therapweqwe y. Ask questions about dosing, efficacy,
                safety, or trial data.
              </p>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-red-800 mb-1">Error</h3>
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            </div>
          )}

          <div className="space-y-6">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && <LoadingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </main>

      <ChatInput
        onSend={handleSendMessage}
        disabled={isLoading || !openAIService}
      />
    </div>
  );
}

export default App;
