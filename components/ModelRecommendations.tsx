import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  CpuChipIcon,
  ClipboardDocumentIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';
import { Recommendations, ModelRecommendation } from '../lib/llmDatabase';
import { SystemInfo } from '../lib/systemAnalyzer';

interface ModelRecommendationsProps {
  recommendations: Recommendations;
  systemInfo: SystemInfo;
}

interface ModelCardProps {
  model: ModelRecommendation;
  category: 'excellent' | 'good' | 'basic';
  index: number;
}

const ModelCard: React.FC<ModelCardProps> = ({ model, category, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const performanceColors = {
    excellent: {
      bg: 'bg-green-50 border-green-200',
      badge: 'bg-green-500 text-white',
      text: 'text-green-800'
    },
    good: {
      bg: 'bg-yellow-50 border-yellow-200',
      badge: 'bg-yellow-500 text-white',
      text: 'text-yellow-800'
    },
    basic: {
      bg: 'bg-orange-50 border-orange-200',
      badge: 'bg-orange-500 text-white',
      text: 'text-orange-800'
    }
  };

  const colors = performanceColors[category];

  const copyToClipboard = (text: string, type: string) => {
    navigator.clipboard.writeText(text).then(() => {
      toast.success(`${type} copied to clipboard!`);
    }).catch(() => {
      toast.error('Failed to copy to clipboard');
    });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className={`${colors.bg} border rounded-xl p-6 hover:shadow-lg transition-all duration-300`}
    >
      {/* Model Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center flex-wrap gap-3">
          <h3 className="text-xl font-semibold text-gray-900">{model.name}</h3>
          {model.specs.domain && (
            <span className="bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
              {model.specs.domain}
            </span>
          )}
        </div>
        <span className={`${colors.badge} px-3 py-1 rounded-full text-sm font-medium`}>
          {model.compatibility.performance_tier}
        </span>
      </div>

      {/* Model Description */}
      <p className="text-gray-700 mb-4 leading-relaxed">{model.specs.description}</p>

      {/* Requirements Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
        <div className="bg-white rounded-lg p-3 text-center border">
          <div className="text-xs text-gray-500 mb-1">Parameters</div>
          <div className="font-semibold text-gray-900">{model.specs.parameters}</div>
        </div>
        <div className="bg-white rounded-lg p-3 text-center border">
          <div className="text-xs text-gray-500 mb-1">Min RAM</div>
          <div className="font-semibold text-gray-900">{model.specs.min_ram_gb} GB</div>
        </div>
        <div className="bg-white rounded-lg p-3 text-center border">
          <div className="text-xs text-gray-500 mb-1">Rec. RAM</div>
          <div className="font-semibold text-gray-900">{model.specs.recommended_ram_gb} GB</div>
        </div>
        <div className="bg-white rounded-lg p-3 text-center border">
          <div className="text-xs text-gray-500 mb-1">Min VRAM</div>
          <div className="font-semibold text-gray-900">{model.specs.min_vram_gb} GB</div>
        </div>
        <div className="bg-white rounded-lg p-3 text-center border">
          <div className="text-xs text-gray-500 mb-1">Rec. VRAM</div>
          <div className="font-semibold text-gray-900">{model.specs.recommended_vram_gb} GB</div>
        </div>
      </div>

      {/* Compatibility Notes */}
      {(model.compatibility.recommended_quant || model.compatibility.notes.length > 0) && (
        <div className="mb-4 space-y-2">
          {model.compatibility.recommended_quant && (
            <div className="flex items-center text-sm">
              <CheckCircleIcon className="h-4 w-4 text-green-500 mr-2" />
              <span className="text-gray-700">
                <strong>Recommended quantization:</strong> {model.compatibility.recommended_quant}
              </span>
            </div>
          )}
          {model.compatibility.notes.map((note, noteIndex) => (
            <div key={noteIndex} className="flex items-center text-sm">
              <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500 mr-2" />
              <span className="text-gray-700">{note}</span>
            </div>
          ))}
        </div>
      )}

      {/* Installation Methods Toggle */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between bg-white rounded-lg p-3 border hover:bg-gray-50 transition-colors"
      >
        <span className="font-medium text-gray-900">Installation Instructions</span>
        {isExpanded ? (
          <ChevronUpIcon className="h-5 w-5 text-gray-500" />
        ) : (
          <ChevronDownIcon className="h-5 w-5 text-gray-500" />
        )}
      </button>

      {/* Installation Methods */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 space-y-4"
          >
            {/* Ollama */}
            {model.specs.install_methods.ollama && (
              <div className="bg-white rounded-lg p-4 border">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold text-gray-900">📱 Ollama (Recommended)</h4>
                  <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium">
                    Easy
                  </span>
                </div>
                <div className="bg-gray-900 rounded-lg p-3 relative mb-3">
                  <code className="text-green-400 text-sm">
                    $ {model.specs.install_methods.ollama.command}
                  </code>
                  <button
                    onClick={() => copyToClipboard(model.specs.install_methods.ollama!.command!, 'Command')}
                    className="absolute top-2 right-2 p-1 hover:bg-gray-700 rounded"
                  >
                    <ClipboardDocumentIcon className="h-4 w-4 text-gray-400" />
                  </button>
                </div>
                <p className="text-sm text-gray-600">{model.specs.install_methods.ollama.note}</p>
              </div>
            )}

            {/* HuggingFace */}
            {model.specs.install_methods.huggingface && (
              <div className="bg-white rounded-lg p-4 border">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold text-gray-900">🤗 HuggingFace</h4>
                  <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-xs font-medium">
                    Intermediate
                  </span>
                </div>
                <div className="mb-3">
                  <div className="flex items-center justify-between bg-gray-50 rounded-lg p-2">
                    <code className="text-sm text-gray-700 flex-1">
                      {model.specs.install_methods.huggingface.model_id}
                    </code>
                    <button
                      onClick={() => copyToClipboard(model.specs.install_methods.huggingface!.model_id!, 'Model ID')}
                      className="ml-2 p-1 hover:bg-gray-200 rounded"
                    >
                      <ClipboardDocumentIcon className="h-4 w-4 text-gray-500" />
                    </button>
                  </div>
                </div>
                <p className="text-sm text-gray-600">{model.specs.install_methods.huggingface.note}</p>
              </div>
            )}

            {/* GGUF */}
            {model.specs.install_methods.gguf && (
              <div className="bg-white rounded-lg p-4 border">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold text-gray-900">⚙️ GGUF (llama.cpp)</h4>
                  <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs font-medium">
                    Advanced
                  </span>
                </div>
                <div className="mb-3">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Download Source:</label>
                  <a 
                    href={model.specs.install_methods.gguf.source}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 text-sm break-all"
                  >
                    {model.specs.install_methods.gguf.source}
                  </a>
                </div>
                <div className="mb-2">
                  <span className="text-sm font-medium text-gray-700">Recommended:</span>
                  <span className="ml-2 text-sm text-gray-600">{model.specs.install_methods.gguf.recommended_quant}</span>
                </div>
                <p className="text-sm text-gray-600">{model.specs.install_methods.gguf.note}</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const ModelRecommendations: React.FC<ModelRecommendationsProps> = ({ 
  recommendations, 
  systemInfo 
}) => {
  const [activeTab, setActiveTab] = useState<'all' | 'general' | 'specialized'>('all');

  const suitableModels = [
    ...recommendations.excellent,
    ...recommendations.good,
    ...recommendations.basic
  ];

  const generalModels = suitableModels.filter(model => !model.specs.domain);
  const specializedModels = suitableModels.filter(model => model.specs.domain);

  const getModelsToShow = () => {
    switch (activeTab) {
      case 'general':
        return { 
          excellent: recommendations.excellent.filter(m => !m.specs.domain),
          good: recommendations.good.filter(m => !m.specs.domain),
          basic: recommendations.basic.filter(m => !m.specs.domain)
        };
      case 'specialized':
        return { 
          excellent: recommendations.excellent.filter(m => m.specs.domain),
          good: recommendations.good.filter(m => m.specs.domain),
          basic: recommendations.basic.filter(m => m.specs.domain)
        };
      default:
        return recommendations;
    }
  };

  const modelsToShow = getModelsToShow();

  if (suitableModels.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden"
      >
        <div className="bg-gradient-to-r from-red-500 to-pink-500 px-6 py-4">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-8 w-8 text-white mr-3" />
            <h2 className="text-2xl font-bold text-white">Insufficient Hardware</h2>
          </div>
        </div>
        
        <div className="p-6">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold text-red-800 mb-4">System Requirements Not Met</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div>
                <span className="text-red-600 font-medium">RAM:</span>
                <p className="text-red-800">{systemInfo.totalRamGB} GB</p>
              </div>
              <div>
                <span className="text-red-600 font-medium">Storage:</span>
                <p className="text-red-800">{systemInfo.freeStorageGB} GB free</p>
              </div>
              <div>
                <span className="text-red-600 font-medium">GPUs:</span>
                <p className="text-red-800">{systemInfo.gpus?.length || 0} detected</p>
              </div>
            </div>
            <p className="text-red-700">
              Your system doesn't meet the minimum requirements for running local LLMs efficiently.
            </p>
          </div>

          <h3 className="text-xl font-semibold text-gray-800 mb-4">🌐 Recommended Cloud-Based Solutions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { name: 'ChatGPT', url: 'https://chat.openai.com', desc: 'OpenAI\'s flagship conversational AI' },
              { name: 'Claude', url: 'https://claude.ai', desc: 'Anthropic\'s helpful AI assistant' },
              { name: 'Google Bard', url: 'https://bard.google.com', desc: 'Google\'s experimental AI service' },
              { name: 'Perplexity AI', url: 'https://perplexity.ai', desc: 'AI-powered search and Q&A' }
            ].map((service, index) => (
              <a
                key={index}
                href={service.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block bg-blue-50 border border-blue-200 rounded-lg p-4 hover:bg-blue-100 transition-colors"
              >
                <h4 className="font-semibold text-blue-900 mb-2">{service.name}</h4>
                <p className="text-blue-700 text-sm">{service.desc}</p>
              </a>
            ))}
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden"
    >
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <CpuChipIcon className="h-8 w-8 text-white mr-3" />
            <h2 className="text-2xl font-bold text-white">Model Recommendations</h2>
          </div>
          <div className="text-white text-sm">
            {suitableModels.length} compatible models found
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-green-50 rounded-lg p-4 text-center border border-green-200">
            <div className="text-2xl font-bold text-green-600 mb-1">
              {recommendations.excellent.length}
            </div>
            <div className="text-sm text-green-700">Excellent</div>
          </div>
          <div className="bg-yellow-50 rounded-lg p-4 text-center border border-yellow-200">
            <div className="text-2xl font-bold text-yellow-600 mb-1">
              {recommendations.good.length}
            </div>
            <div className="text-sm text-yellow-700">Good</div>
          </div>
          <div className="bg-orange-50 rounded-lg p-4 text-center border border-orange-200">
            <div className="text-2xl font-bold text-orange-600 mb-1">
              {recommendations.basic.length}
            </div>
            <div className="text-sm text-orange-700">Basic</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4 text-center border border-gray-200">
            <div className="text-2xl font-bold text-gray-600 mb-1">
              {recommendations.not_suitable.length}
            </div>
            <div className="text-sm text-gray-700">Not Suitable</div>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
          {[
            { key: 'all', label: 'All Models', count: suitableModels.length },
            { key: 'general', label: 'General Purpose', count: generalModels.length },
            { key: 'specialized', label: 'Specialized', count: specializedModels.length }
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key as any)}
              className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.key
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {tab.label} ({tab.count})
            </button>
          ))}
        </div>

        {/* Model Categories */}
        {[
          { key: 'excellent', title: '🟢 Excellent Performance', models: modelsToShow.excellent },
          { key: 'good', title: '🟡 Good Performance', models: modelsToShow.good },
          { key: 'basic', title: '🟠 Basic Performance', models: modelsToShow.basic }
        ].map(({ key, title, models }) => (
          models.length > 0 && (
            <div key={key} className="mb-8">
              <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                {title}
                {key === 'excellent' && specializedModels.some(m => modelsToShow.excellent.includes(m)) && (
                  <BeakerIcon className="h-5 w-5 text-purple-500 ml-2" title="Includes specialized models" />
                )}
              </h3>
              <div className="space-y-4">
                {models.map((model, index) => (
                  <ModelCard
                    key={model.name}
                    model={model}
                    category={key as any}
                    index={index}
                  />
                ))}
              </div>
            </div>
          )
        ))}
      </div>
    </motion.div>
  );
};

export default ModelRecommendations;