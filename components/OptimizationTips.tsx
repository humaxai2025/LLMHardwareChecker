import React from 'react';
import { motion } from 'framer-motion';
import {
  LightBulbIcon,
  CpuChipIcon,
  CircleStackIcon,
  ServerStackIcon,
  DevicePhoneMobileIcon,
  ComputerDesktopIcon,
  BoltIcon
} from '@heroicons/react/24/outline';
import { LLMRecommender } from '../lib/llmRecommender';

interface OptimizationTipsProps {
  recommender: LLMRecommender;
}

const OptimizationTips: React.FC<OptimizationTipsProps> = ({ recommender }) => {
  const tips = recommender.getOptimizationTips();
  const capabilityLevel = recommender.getSystemCapabilityLevel();

  // Categorize tips
  const categorizedTips = {
    memory: tips.filter(tip => tip.includes('RAM') || tip.includes('memory') || tip.includes('Memory')),
    gpu: tips.filter(tip => tip.includes('GPU') || tip.includes('VRAM') || tip.includes('Metal') || tip.includes('Apple Silicon')),
    storage: tips.filter(tip => tip.includes('storage') || tip.includes('disk') || tip.includes('models')),
    performance: tips.filter(tip => tip.includes('quantization') || tip.includes('CPU') || tip.includes('cores') || tip.includes('Q4_K_M')),
    platform: tips.filter(tip => tip.includes('Windows') || tip.includes('macOS') || tip.includes('Linux') || tip.includes('Mobile')),
    general: tips.filter(tip => 
      !tip.includes('RAM') && !tip.includes('GPU') && !tip.includes('storage') && 
      !tip.includes('quantization') && !tip.includes('Windows') && !tip.includes('macOS') && 
      !tip.includes('Linux') && !tip.includes('Metal') && !tip.includes('Apple') && 
      !tip.includes('Mobile') && !tip.includes('VRAM') && !tip.includes('disk')
    )
  };

  const tipCategories = [
    {
      title: 'Memory Optimization',
      icon: CircleStackIcon,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      tips: categorizedTips.memory
    },
    {
      title: 'GPU & Graphics',
      icon: CpuChipIcon,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200',
      tips: categorizedTips.gpu
    },
    {
      title: 'Performance Tuning',
      icon: BoltIcon,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-200',
      tips: categorizedTips.performance
    },
    {
      title: 'Storage Management',
      icon: ServerStackIcon,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      tips: categorizedTips.storage
    },
    {
      title: 'Platform Specific',
      icon: ComputerDesktopIcon,
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-50',
      borderColor: 'border-indigo-200',
      tips: categorizedTips.platform
    },
    {
      title: 'General Tips',
      icon: LightBulbIcon,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      borderColor: 'border-orange-200',
      tips: categorizedTips.general
    }
  ].filter(category => category.tips.length > 0);

  const capabilityLevelConfig = {
    low: {
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      borderColor: 'border-orange-200',
      label: 'Entry Level System',
      description: 'Focus on lightweight models and CPU optimization'
    },
    medium: {
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      label: 'Mid-Range System',
      description: 'Can handle 7B models with good performance'
    },
    high: {
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      label: 'High-End System',
      description: 'Capable of running large models efficiently'
    },
    premium: {
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200',
      label: 'Premium System',
      description: 'Can run the largest models with excellent performance'
    }
  };

  const currentCapability = capabilityLevelConfig[capabilityLevel];

  const formatTip = (tip: string) => {
    // Remove emoji from the beginning if present
    return tip.replace(/^[🔧💾🎮💻🍎🐧🪟📱⚡🗑️❌✅🚀💡⚙️🔋☁️📁]+\s*/, '');
  };

  const getEmojiForTip = (tip: string) => {
    const match = tip.match(/^([🔧💾🎮💻🍎🐧🪟📱⚡🗑️❌✅🚀💡⚙️🔋☁️📁]+)/);
    return match ? match[1] : '💡';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden"
    >
      <div className="bg-gradient-to-r from-yellow-500 to-orange-500 px-6 py-4">
        <div className="flex items-center">
          <LightBulbIcon className="h-8 w-8 text-white mr-3" />
          <h2 className="text-2xl font-bold text-white">Optimization Tips for Your System</h2>
        </div>
      </div>

      <div className="p-6">
        {/* System Capability Level */}
        <div className={`${currentCapability.bgColor} ${currentCapability.borderColor} border rounded-lg p-6 mb-8`}>
          <div className="flex items-center mb-3">
            <ComputerDesktopIcon className={`h-6 w-6 ${currentCapability.color} mr-3`} />
            <h3 className={`text-lg font-semibold ${currentCapability.color}`}>
              System Capability: {currentCapability.label}
            </h3>
          </div>
          <p className={`${currentCapability.color} opacity-80`}>
            {currentCapability.description}
          </p>
        </div>

        {/* Tips Categories */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {tipCategories.map((category, categoryIndex) => (
            <motion.div
              key={category.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: categoryIndex * 0.1 }}
              className={`${category.bgColor} ${category.borderColor} border rounded-lg p-6`}
            >
              <div className="flex items-center mb-4">
                <category.icon className={`h-6 w-6 ${category.color} mr-3`} />
                <h3 className={`text-lg font-semibold ${category.color}`}>
                  {category.title}
                </h3>
              </div>
              
              <div className="space-y-3">
                {category.tips.map((tip, tipIndex) => (
                  <motion.div
                    key={tipIndex}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: (categoryIndex * 0.1) + (tipIndex * 0.05) }}
                    className="flex items-start bg-white rounded-lg p-3 border border-gray-200 shadow-sm"
                  >
                    <span className="text-lg mr-3 flex-shrink-0">
                      {getEmojiForTip(tip)}
                    </span>
                    <p className="text-gray-700 text-sm leading-relaxed">
                      {formatTip(tip)}
                    </p>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Performance Recommendations Based on System Level */}
        <div className="mt-8 bg-gray-50 border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <BoltIcon className="h-6 w-6 text-gray-600 mr-3" />
            Performance Recommendations for Your System
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-700 mb-3">Recommended Models</h4>
              <div className="space-y-2">
                {capabilityLevel === 'low' && (
                  <>
                    <div className="flex items-center text-sm text-gray-600">
                      <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                      Gemma 2B, Phi-3 Mini (3.8B)
                    </div>
                    <div className="flex items-center text-sm text-gray-600">
                      <span className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
                      Llama 3.2 3B (with quantization)
                    </div>
                  </>
                )}
                {capabilityLevel === 'medium' && (
                  <>
                    <div className="flex items-center text-sm text-gray-600">
                      <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                      Llama 3.1 8B, Mistral 7B
                    </div>
                    <div className="flex items-center text-sm text-gray-600">
                      <span className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
                      Code Llama 7B, StarCoder 7B
                    </div>
                  </>
                )}
                {capabilityLevel === 'high' && (
                  <>
                    <div className="flex items-center text-sm text-gray-600">
                      <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                      Llama 3.1 13B, Vicuna 13B
                    </div>
                    <div className="flex items-center text-sm text-gray-600">
                      <span className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
                      Nous Hermes 2 Solar 10.7B
                    </div>
                  </>
                )}
                {capabilityLevel === 'premium' && (
                  <>
                    <div className="flex items-center text-sm text-gray-600">
                      <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                      Llama 3.1 70B, Code Llama 34B
                    </div>
                    <div className="flex items-center text-sm text-gray-600">
                      <span className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
                      Any model with optimal quantization
                    </div>
                  </>
                )}
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-gray-700 mb-3">Optimal Settings</h4>
              <div className="space-y-2">
                {capabilityLevel === 'low' && (
                  <>
                    <div className="text-sm text-gray-600">• Use Q3_K_M or Q4_K_M quantization</div>
                    <div className="text-sm text-gray-600">• Close unnecessary applications</div>
                    <div className="text-sm text-gray-600">• Consider CPU-only inference</div>
                  </>
                )}
                {capabilityLevel === 'medium' && (
                  <>
                    <div className="text-sm text-gray-600">• Use Q4_K_M quantization</div>
                    <div className="text-sm text-gray-600">• Enable GPU acceleration if available</div>
                    <div className="text-sm text-gray-600">• Monitor memory usage</div>
                  </>
                )}
                {capabilityLevel === 'high' && (
                  <>
                    <div className="text-sm text-gray-600">• Use Q5_K_M or Q8_0 quantization</div>
                    <div className="text-sm text-gray-600">• GPU acceleration recommended</div>
                    <div className="text-sm text-gray-600">• Can run multiple models</div>
                  </>
                )}
                {capabilityLevel === 'premium' && (
                  <>
                    <div className="text-sm text-gray-600">• Use Q8_0 or FP16 for best quality</div>
                    <div className="text-sm text-gray-600">• Full GPU utilization</div>
                    <div className="text-sm text-gray-600">• Concurrent model serving possible</div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Quick Action Items */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-800 mb-4">🎯 Next Steps</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white rounded-lg p-4 border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-2">1. Choose Platform</h4>
              <p className="text-blue-700 text-sm">Start with Ollama for easiest setup</p>
            </div>
            <div className="bg-white rounded-lg p-4 border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-2">2. Select Model</h4>
              <p className="text-blue-700 text-sm">Pick a model that fits your system capabilities</p>
            </div>
            <div className="bg-white rounded-lg p-4 border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-2">3. Optimize</h4>
              <p className="text-blue-700 text-sm">Apply the tips above for best performance</p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default OptimizationTips;