import { Loader2 } from 'lucide-react';

export function LoadingIndicator() {
  return (
    <div className="flex items-center gap-3 text-slate-600">
      <div className="flex-shrink-0 w-10 h-10 rounded-full bg-emerald-600 flex items-center justify-center">
        <Loader2 className="w-5 h-5 text-white animate-spin" />
      </div>
      <div className="flex-1">
        <div className="inline-block px-6 py-4 rounded-lg bg-white border border-slate-200">
          <p className="text-slate-600">Processing your question...</p>
        </div>
      </div>
    </div>
  );
}
