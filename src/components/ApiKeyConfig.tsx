import { useState } from 'react';
import { Copy, CheckCircle, AlertCircle } from 'lucide-react';

interface ApiKeyConfigProps {
  onApiKeySet: (apiKey: string, assistantId: string) => void;
  isConfigured: boolean;
}

export function ApiKeyConfig({ onApiKeySet, isConfigured }: ApiKeyConfigProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [assistantId, setAssistantId] = useState(
    import.meta.env.VITE_ASSISTANT_ID || 'asst_zvFuLn2wRlDVEh5SFO2jYsey'
  );
  const [copied, setCopied] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (apiKey.trim()) {
      onApiKeySet(apiKey, assistantId);
      setApiKey('');
      setIsOpen(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(apiKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`ml-auto px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
          isConfigured
            ? 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200'
            : 'bg-amber-100 text-amber-700 hover:bg-amber-200'
        }`}
      >
        {isConfigured ? 'API Key Configured' : 'Configure API Key'}
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="px-6 py-4 border-b border-slate-200">
              <h2 className="text-lg font-semibold text-slate-800">
                Configure OpenAI API Key
              </h2>
            </div>

            <form onSubmit={handleSubmit} className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  API Key
                </label>
                <div className="relative">
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-..."
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                  />
                  {apiKey && (
                    <button
                      type="button"
                      onClick={handleCopy}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-700"
                      title="Copy API key"
                    >
                      {copied ? (
                        <CheckCircle className="w-5 h-5 text-emerald-600" />
                      ) : (
                        <Copy className="w-5 h-5" />
                      )}
                    </button>
                  )}
                </div>
                <p className="mt-2 text-xs text-slate-600">
                  Get your API key from{' '}
                  <a
                    href="https://platform.openai.com/api-keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-emerald-600 hover:text-emerald-700 underline"
                  >
                    platform.openai.com/api-keys
                  </a>
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Assistant ID
                </label>
                <input
                  type="text"
                  value={assistantId}
                  onChange={(e) => setAssistantId(e.target.value)}
                  placeholder="asst_..."
                  className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                />
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setIsOpen(false)}
                  className="flex-1 px-4 py-2 border border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors disabled:bg-slate-300"
                  disabled={!apiKey.trim()}
                >
                  Configure
                </button>
              </div>
            </form>

            <div className="px-6 py-3 bg-blue-50 border-t border-slate-200 flex gap-2 text-xs text-blue-700 rounded-b-lg">
              <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <p>
                Your API key is stored only in your browser and never sent to
                our servers.
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
