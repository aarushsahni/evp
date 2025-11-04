import { Message } from '../types';
import { formatMarkdown } from '../utils/markdown';
import { User, Stethoscope } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isUser ? 'bg-slate-600' : 'bg-emerald-600'
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Stethoscope className="w-5 h-5 text-white" />
        )}
      </div>
      <div
        className={`flex-1 max-w-3xl ${isUser ? 'text-right' : 'text-left'}`}
      >
        <div
          className={`inline-block px-6 py-4 rounded-lg ${
            isUser
              ? 'bg-slate-600 text-white'
              : 'bg-white border border-slate-200'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div
              className="prose prose-sm max-w-none"
              dangerouslySetInnerHTML={{
                __html: formatMarkdown(message.content),
              }}
            />
          )}
        </div>
        <div
          className={`mt-1 text-xs text-slate-500 ${isUser ? 'text-right' : 'text-left'}`}
        >
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}
