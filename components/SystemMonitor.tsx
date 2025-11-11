'use client';

import React, { useState, useEffect } from 'react';
import {
  CommandLineIcon,
  ServerIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline';

interface OllamaModel {
  name: string;
  size: number;
  modified_at: string;
}

interface SystemMonitorProps {
  totalRamGB: number;
  gpuCount: number;
}

const SystemMonitor: React.FC<SystemMonitorProps> = () => {
  const [isOllamaRunning, setIsOllamaRunning] = useState(false);
  const [lmStudioRunning, setLmStudioRunning] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    // Check for Ollama
    const checkOllama = async () => {
      try {
        const response = await fetch('http://localhost:11434/api/tags', {
          method: 'GET',
        });
        if (response.ok) {
          const data = await response.json();
          setIsOllamaRunning(true);
          setOllamaModels(data.models || []);
        } else {
          setIsOllamaRunning(false);
          setOllamaModels([]);
        }
      } catch {
        setIsOllamaRunning(false);
        setOllamaModels([]);
      }
    };

    // Check for LM Studio
    const checkLMStudio = async () => {
      try {
        const response = await fetch('http://localhost:1234/v1/models', {
          method: 'GET',
        });
        setLmStudioRunning(response.ok);
      } catch {
        setLmStudioRunning(false);
      }
    };

    checkOllama();
    checkLMStudio();
    setIsChecking(false);

    const checkInterval = setInterval(() => {
      checkOllama();
      checkLMStudio();
    }, 10000); // Check every 10 seconds

    return () => {
      clearInterval(checkInterval);
    };
  }, []);

  return (
    <div className="space-y-4">
      {/* LLM Runner Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div
          className={`flex items-center gap-3 p-4 rounded-lg border-2 transition-all ${
            isOllamaRunning
              ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
              : 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-700'
          }`}
        >
          <div
            className={`w-3 h-3 rounded-full ${
              isOllamaRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
            }`}
          />
          <CommandLineIcon className="h-6 w-6 text-gray-600 dark:text-gray-400" />
          <div className="flex-1">
            <div className="text-base font-semibold text-gray-900 dark:text-gray-100">Ollama</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {isChecking ? 'Checking...' : isOllamaRunning ? 'Running on :11434' : 'Not detected'}
            </div>
            {isOllamaRunning && ollamaModels.length > 0 && (
              <div className="text-xs text-green-600 dark:text-green-400 mt-1">
                {ollamaModels.length} model{ollamaModels.length !== 1 ? 's' : ''} available
              </div>
            )}
          </div>
          {isOllamaRunning ? (
            <CheckCircleIcon className="h-6 w-6 text-green-500" />
          ) : (
            <XCircleIcon className="h-6 w-6 text-gray-400" />
          )}
        </div>

        <div
          className={`flex items-center gap-3 p-4 rounded-lg border-2 transition-all ${
            lmStudioRunning
              ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
              : 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-700'
          }`}
        >
          <div
            className={`w-3 h-3 rounded-full ${
              lmStudioRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
            }`}
          />
          <ServerIcon className="h-6 w-6 text-gray-600 dark:text-gray-400" />
          <div className="flex-1">
            <div className="text-base font-semibold text-gray-900 dark:text-gray-100">LM Studio</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {isChecking ? 'Checking...' : lmStudioRunning ? 'Running on :1234' : 'Not detected'}
            </div>
          </div>
          {lmStudioRunning ? (
            <CheckCircleIcon className="h-6 w-6 text-green-500" />
          ) : (
            <XCircleIcon className="h-6 w-6 text-gray-400" />
          )}
        </div>
      </div>

      {/* Ollama Models List */}
      {isOllamaRunning && ollamaModels.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
            Installed Ollama Models
          </h4>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {ollamaModels.map((model, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900/50 rounded border border-gray-200 dark:border-gray-700"
              >
                <div className="flex-1">
                  <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {model.name}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {(model.size / (1024 * 1024 * 1024)).toFixed(2)} GB
                  </div>
                </div>
                <CheckCircleIcon className="h-5 w-5 text-green-500" />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info Note */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          <strong>Local LLM Detection:</strong> This tool checks for Ollama (localhost:11434) and LM Studio
          (localhost:1234) running on your machine. Start these services to see them detected here.
        </p>
      </div>
    </div>
  );
};

export default SystemMonitor;
