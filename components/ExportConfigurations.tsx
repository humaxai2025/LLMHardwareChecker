'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  DocumentArrowDownIcon,
  ClipboardDocumentIcon,
  CodeBracketIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';
import { ModelRecommendation } from '../lib/llmDatabase';
import { SystemInfo } from '../lib/systemAnalyzer';

interface ExportConfigurationsProps {
  model: ModelRecommendation;
  systemInfo: SystemInfo;
}

type ExportType = 'docker' | 'modelfile' | 'llamacpp' | 'api';

const ExportConfigurations: React.FC<ExportConfigurationsProps> = ({ model, systemInfo }) => {
  const [selectedExport, setSelectedExport] = useState<ExportType>('docker');
  const [copied, setCopied] = useState(false);

  const hasGPU = systemInfo.gpus && systemInfo.gpus.length > 0;
  const gpuVram = hasGPU ? (typeof systemInfo.gpus![0].vramGB === 'number'
    ? systemInfo.gpus![0].vramGB
    : parseInt(systemInfo.gpus![0].vramGB as string) || 0) : 0;

  const generateDockerCompose = () => {
    return `version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-${model.name.toLowerCase().replace(/\s+/g, '-')}
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
${hasGPU ? `    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]` : ''}
    restart: unless-stopped
    command: serve

  # Auto-pull model on startup
  model-setup:
    image: ollama/ollama:latest
    depends_on:
      - ollama
    entrypoint: >
      sh -c "sleep 10 && ollama pull ${model.specs.install_methods.ollama?.command?.replace('ollama run ', '') || model.name.toLowerCase()}"
    restart: on-failure

volumes:
  ollama-data:
    driver: local

# Usage:
# docker-compose up -d
# docker exec -it ollama-${model.name.toLowerCase().replace(/\s+/g, '-')} ollama run ${model.specs.install_methods.ollama?.command?.replace('ollama run ', '') || model.name.toLowerCase()}`;
  };

  const generateModelfile = () => {
    const threads = Math.min(systemInfo.cpuCores, 8);
    const contextLength = model.specs.min_ram_gb <= 8 ? 2048 : model.specs.min_ram_gb <= 16 ? 4096 : 8192;

    return `# Modelfile for ${model.name}
FROM ${model.specs.install_methods.ollama?.command?.replace('ollama run ', '') || model.name.toLowerCase()}

# Hardware-optimized parameters for your system
PARAMETER num_ctx ${contextLength}
PARAMETER num_thread ${threads}
PARAMETER num_gpu ${hasGPU ? Math.min(gpuVram >= 8 ? 99 : 35, 99) : 0}
PARAMETER num_batch ${hasGPU ? 512 : 128}
${hasGPU ? 'PARAMETER use_mmap true' : 'PARAMETER use_mmap false'}
${hasGPU ? 'PARAMETER use_mlock true' : ''}

# Performance tuning
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt
SYSTEM """
You are a helpful, harmless, and honest AI assistant.
"""

# Usage:
# ollama create ${model.name.toLowerCase().replace(/\s+/g, '-')}-optimized -f ./Modelfile
# ollama run ${model.name.toLowerCase().replace(/\s+/g, '-')}-optimized`;
  };

  const generateLlamaCppCommand = () => {
    const threads = Math.min(systemInfo.cpuCores - 1, 8);
    const gpuLayers = hasGPU ? (gpuVram >= 8 ? 99 : 35) : 0;

    return `# llama.cpp command optimized for your hardware
# Download model from HuggingFace first

./main \\
  -m ./models/${model.name.toLowerCase().replace(/\s+/g, '-')}.gguf \\
  -t ${threads} \\
  -c ${model.specs.min_ram_gb <= 8 ? 2048 : 4096} \\
  -b ${hasGPU ? 512 : 128} \\
${hasGPU ? `  -ngl ${gpuLayers} \\\n` : ''}  --temp 0.7 \\
  --top-p 0.9 \\
  --top-k 40 \\
  --repeat-penalty 1.1 \\
  -p "### System: You are a helpful assistant.\\n\\n### User: Hello!\\n\\n### Assistant:"

# Flags explained:
# -t: CPU threads (${threads})
# -c: Context length (${model.specs.min_ram_gb <= 8 ? 2048 : 4096})
# -b: Batch size (${hasGPU ? 512 : 128})
${hasGPU ? `# -ngl: GPU layers (${gpuLayers})\n` : ''}# --temp: Temperature for sampling (0.7)
# -p: Prompt template

# For interactive mode:
# ./main -m ./models/${model.name.toLowerCase().replace(/\s+/g, '-')}.gguf -t ${threads} -c ${model.specs.min_ram_gb <= 8 ? 2048 : 4096}${hasGPU ? ` -ngl ${gpuLayers}` : ''} --interactive`;
  };

  const generateAPIConfig = () => {
    return `# OpenAI-Compatible API Configuration

## Using with Ollama
# Start Ollama server:
ollama serve

# Set environment variable:
export OLLAMA_HOST=http://localhost:11434

# Python Example:
\`\`\`python
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # required but unused
)

response = client.chat.completions.create(
    model="${model.specs.install_methods.ollama?.command?.replace('ollama run ', '') || model.name.toLowerCase()}",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=${model.specs.min_ram_gb <= 8 ? 512 : 1024}
)

print(response.choices[0].message.content)
\`\`\`

## Using with LM Studio
# Start LM Studio server (default port 1234)
# Base URL: http://localhost:1234/v1

## cURL Example:
curl http://localhost:11434/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${model.specs.install_methods.ollama?.command?.replace('ollama run ', '') || model.name.toLowerCase()}",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": ${model.specs.min_ram_gb <= 8 ? 512 : 1024}
  }'

## Environment Variables:
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODELS=/path/to/models
# OLLAMA_NUM_PARALLEL=1
# OLLAMA_MAX_LOADED_MODELS=1`;
  };

  const getConfigContent = () => {
    switch (selectedExport) {
      case 'docker':
        return generateDockerCompose();
      case 'modelfile':
        return generateModelfile();
      case 'llamacpp':
        return generateLlamaCppCommand();
      case 'api':
        return generateAPIConfig();
      default:
        return '';
    }
  };

  const copyToClipboard = () => {
    const content = getConfigContent();
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      toast.success('Configuration copied to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const downloadConfig = () => {
    const content = getConfigContent();
    const extension = selectedExport === 'docker' ? 'yml' :
                     selectedExport === 'modelfile' ? 'Modelfile' :
                     selectedExport === 'llamacpp' ? 'sh' : 'md';
    const filename = `${model.name.toLowerCase().replace(/\s+/g, '-')}-${selectedExport}.${extension}`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Configuration downloaded as ${filename}`);
  };

  const exportTypes = [
    { key: 'docker' as ExportType, label: 'Docker Compose', icon: '🐳' },
    { key: 'modelfile' as ExportType, label: 'Ollama Modelfile', icon: '📄' },
    { key: 'llamacpp' as ExportType, label: 'llama.cpp', icon: '⚡' },
    { key: 'api' as ExportType, label: 'API Config', icon: '🔌' },
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-indigo-700 px-6 py-4 border-b border-indigo-800">
        <div className="flex items-center">
          <div className="p-2 bg-white bg-opacity-20 rounded-lg mr-3">
            <CodeBracketIcon className="h-6 w-6 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Export Configuration</h3>
            <p className="text-indigo-100 text-sm">Generate optimized configs for {model.name}</p>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Export Type Selector */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          {exportTypes.map((type) => (
            <button
              key={type.key}
              onClick={() => setSelectedExport(type.key)}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedExport === type.key
                  ? 'border-indigo-600 bg-indigo-50 dark:bg-indigo-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-indigo-300 dark:hover:border-indigo-700'
              }`}
            >
              <div className="text-2xl mb-2">{type.icon}</div>
              <div className="text-sm font-medium text-gray-900 dark:text-gray-100">{type.label}</div>
            </button>
          ))}
        </div>

        {/* Config Content */}
        <div className="bg-gray-900 rounded-lg p-4 mb-4 overflow-x-auto">
          <pre className="text-sm text-gray-100 font-mono whitespace-pre-wrap">{getConfigContent()}</pre>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={copyToClipboard}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 transition-colors"
          >
            {copied ? (
              <>
                <CheckCircleIcon className="h-5 w-5" />
                Copied!
              </>
            ) : (
              <>
                <ClipboardDocumentIcon className="h-5 w-5" />
                Copy to Clipboard
              </>
            )}
          </button>
          <button
            onClick={downloadConfig}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 transition-colors"
          >
            <DocumentArrowDownIcon className="h-5 w-5" />
            Download File
          </button>
        </div>

        {/* Hardware Info */}
        <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <p className="text-xs text-blue-800 dark:text-blue-200">
            <strong>Optimized for your hardware:</strong> {systemInfo.cpuCores} CPU cores, {systemInfo.totalRamGB}GB RAM
            {hasGPU && `, ${gpuVram}GB VRAM`}
          </p>
        </div>
      </div>
    </div>
  );
};

export default ExportConfigurations;
