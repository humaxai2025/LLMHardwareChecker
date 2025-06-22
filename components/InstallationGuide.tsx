import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  WrenchScrewdriverIcon,
  ClipboardDocumentIcon,
  ComputerDesktopIcon,
  CommandLineIcon,
  CodeBracketIcon,
  CubeIcon
} from '@heroicons/react/24/outline';
import { LLMRecommender } from '../lib/llmRecommender';

interface InstallationGuideProps {
  recommender: LLMRecommender;
}

const InstallationGuide: React.FC<InstallationGuideProps> = ({ recommender }) => {
  const [selectedPlatform, setSelectedPlatform] = useState<string>('ollama');
  const platforms = recommender.getInstallationPlatforms();

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      toast.success('Command copied to clipboard!');
    }).catch(() => {
      toast.error('Failed to copy to clipboard');
    });
  };

  const platformIcons = {
    'Ollama': CubeIcon,
    'LM Studio': ComputerDesktopIcon,
    'llama.cpp': CommandLineIcon,
    'HuggingFace Transformers': CodeBracketIcon
  };

  const difficultyColors = {
    'Easy': 'bg-green-100 text-green-800 border-green-200',
    'Medium': 'bg-yellow-100 text-yellow-800 border-yellow-200',
    'Advanced': 'bg-red-100 text-red-800 border-red-200'
  };

  const selectedPlatformData = platforms.find(p => p.name.toLowerCase() === selectedPlatform);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden"
    >
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
        <div className="flex items-center">
          <WrenchScrewdriverIcon className="h-8 w-8 text-white mr-3" />
          <h2 className="text-2xl font-bold text-white">Platform Installation Guide</h2>
        </div>
      </div>

      <div className="p-6">
        {/* Platform Selection */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {platforms.map((platform, index) => {
            const Icon = platformIcons[platform.name as keyof typeof platformIcons] || CubeIcon;
            const isSelected = platform.name.toLowerCase() === selectedPlatform;
            
            return (
              <motion.button
                key={platform.name}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => setSelectedPlatform(platform.name.toLowerCase())}
                className={`p-4 rounded-lg border-2 transition-all duration-300 text-left ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 shadow-md'
                    : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
                }`}
              >
                <div className="flex items-center mb-3">
                  <Icon className={`h-6 w-6 mr-3 ${isSelected ? 'text-blue-600' : 'text-gray-600'}`} />
                  <h3 className={`font-semibold ${isSelected ? 'text-blue-900' : 'text-gray-900'}`}>
                    {platform.name}
                  </h3>
                </div>
                <span className={`inline-block px-2 py-1 rounded text-xs font-medium border ${
                  difficultyColors[platform.difficulty]
                }`}>
                  {platform.difficulty}
                </span>
                <p className={`text-sm mt-2 ${isSelected ? 'text-blue-700' : 'text-gray-600'}`}>
                  {platform.bestFor}
                </p>
              </motion.button>
            );
          })}
        </div>

        {/* Selected Platform Details */}
        {selectedPlatformData && (
          <motion.div
            key={selectedPlatform}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
            className="bg-gray-50 rounded-lg p-6 border border-gray-200"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold text-gray-900 flex items-center">
                {React.createElement(platformIcons[selectedPlatformData.name as keyof typeof platformIcons] || CubeIcon, {
                  className: "h-7 w-7 text-blue-600 mr-3"
                })}
                {selectedPlatformData.name}
              </h3>
              <span className={`px-3 py-1 rounded-full text-sm font-medium border ${
                difficultyColors[selectedPlatformData.difficulty]
              }`}>
                {selectedPlatformData.difficulty} Setup
              </span>
            </div>

            <div className="mb-6">
              <h4 className="text-lg font-semibold text-gray-800 mb-2">Description</h4>
              <p className="text-gray-700 leading-relaxed">{selectedPlatformData.description}</p>
            </div>

            <div className="mb-6">
              <h4 className="text-lg font-semibold text-gray-800 mb-2">Best For</h4>
              <p className="text-gray-700">{selectedPlatformData.bestFor}</p>
            </div>

            <div>
              <h4 className="text-lg font-semibold text-gray-800 mb-4">Installation Instructions</h4>
              <div className="space-y-4">
                {Object.entries(selectedPlatformData.installation).map(([os, instruction]) => (
                  <div key={os} className="bg-white rounded-lg p-4 border border-gray-200">
                    <div className="flex items-center justify-between mb-3">
                      <h5 className="font-semibold text-gray-900 flex items-center">
                        {os === 'Windows' && '🪟'}
                        {os === 'macOS' && '🍎'}
                        {os === 'Linux' && '🐧'}
                        {os === 'All' && '💻'}
                        <span className="ml-2">{os}</span>
                      </h5>
                    </div>
                    
                    {instruction.includes('curl') || instruction.includes('brew') || instruction.includes('pip') ? (
                      <div className="bg-gray-900 rounded-lg p-3 relative">
                        <code className="text-green-400 text-sm block">
                          $ {instruction}
                        </code>
                        <button
                          onClick={() => copyToClipboard(instruction)}
                          className="absolute top-2 right-2 p-1 hover:bg-gray-700 rounded transition-colors"
                          title="Copy command"
                        >
                          <ClipboardDocumentIcon className="h-4 w-4 text-gray-400" />
                        </button>
                      </div>
                    ) : (
                      <p className="text-gray-700 bg-blue-50 p-3 rounded-lg border border-blue-200">
                        {instruction}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Platform-specific tips */}
            <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h5 className="font-semibold text-blue-900 mb-2">💡 Platform Tips</h5>
              <div className="text-blue-800 text-sm space-y-1">
                {selectedPlatform === 'ollama' && (
                  <>
                    <p>• Models are automatically downloaded on first use</p>
                    <p>• Use <code className="bg-blue-100 px-1 rounded">ollama list</code> to see installed models</p>
                    <p>• Use <code className="bg-blue-100 px-1 rounded">ollama pull model-name</code> to download models</p>
                    <p>• Models are stored in ~/.ollama/models/ (macOS/Linux) or %USERPROFILE%\.ollama\models\ (Windows)</p>
                  </>
                )}
                {selectedPlatform === 'lm studio' && (
                  <>
                    <p>• Download models directly through the GUI</p>
                    <p>• Supports both GGUF and HuggingFace models</p>
                    <p>• Built-in chat interface for testing models</p>
                    <p>• Can serve models via API for integration with other apps</p>
                  </>
                )}
                {selectedPlatform === 'llama.cpp' && (
                  <>
                    <p>• Compile with GPU support: <code className="bg-blue-100 px-1 rounded">make LLAMA_CUBLAS=1</code> (NVIDIA)</p>
                    <p>• Use <code className="bg-blue-100 px-1 rounded">-ngl 32</code> to offload layers to GPU</p>
                    <p>• Adjust <code className="bg-blue-100 px-1 rounded">-t</code> parameter for CPU threads</p>
                    <p>• Download GGUF models from HuggingFace for best performance</p>
                  </>
                )}
                {selectedPlatform === 'huggingface transformers' && (
                  <>
                    <p>• Install PyTorch with CUDA support for GPU acceleration</p>
                    <p>• Use <code className="bg-blue-100 px-1 rounded">device_map="auto"</code> for automatic GPU usage</p>
                    <p>• Consider using <code className="bg-blue-100 px-1 rounded">torch_dtype=torch.float16</code> to save memory</p>
                    <p>• Cache models locally to avoid re-downloading</p>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        )}

        {/* Quick Start Guide */}
        <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-yellow-800 mb-4">⚡ Quick Start Recommendation</h3>
          <div className="text-yellow-700 space-y-2">
            <p className="font-medium">For beginners, we recommend starting with Ollama:</p>
            <ol className="list-decimal list-inside space-y-1 ml-4">
              <li>Install Ollama from ollama.ai</li>
              <li>Open terminal/command prompt</li>
              <li>Run: <code className="bg-yellow-100 px-2 py-1 rounded font-mono">ollama run llama3.2:3b</code></li>
              <li>Wait for download and start chatting!</li>
            </ol>
            <p className="text-sm mt-3">
              💡 This will download and run a 3B parameter model that works well on most systems.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default InstallationGuide;