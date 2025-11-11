'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircleIcon,
  XCircleIcon,
  Square2StackIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { LLM_DATABASE, LLMModel } from '../lib/llmDatabase';
import { SystemInfo } from '../lib/systemAnalyzer';
import { PerformanceEstimator } from '../lib/performanceEstimator';

interface ModelComparisonProps {
  systemInfo: SystemInfo;
}

const ModelComparison: React.FC<ModelComparisonProps> = ({ systemInfo }) => {
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  const modelNames = Object.keys(LLM_DATABASE);
  const filteredModels = modelNames.filter(name =>
    name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const toggleModel = (modelName: string) => {
    if (selectedModels.includes(modelName)) {
      setSelectedModels(selectedModels.filter(m => m !== modelName));
    } else if (selectedModels.length < 4) {
      setSelectedModels([...selectedModels, modelName]);
    }
  };

  const clearComparison = () => {
    setSelectedModels([]);
  };

  const performanceEstimator = new PerformanceEstimator(systemInfo);

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Square2StackIcon className="h-8 w-8 text-indigo-500 mr-3" />
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Model Comparison</h2>
            <p className="text-sm text-gray-600">
              Compare up to 4 models side-by-side ({selectedModels.length}/4 selected)
            </p>
          </div>
        </div>
        {selectedModels.length > 0 && (
          <button
            onClick={clearComparison}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Clear All
          </button>
        )}
      </div>

      {/* Model Selector */}
      {selectedModels.length < 4 && (
        <div className="mb-6">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search models to compare..."
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent mb-3"
          />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 max-h-40 overflow-y-auto">
            {filteredModels.map((modelName) => (
              <button
                key={modelName}
                onClick={() => toggleModel(modelName)}
                disabled={selectedModels.includes(modelName)}
                className={`px-3 py-2 text-sm rounded-lg transition-colors ${
                  selectedModels.includes(modelName)
                    ? 'bg-indigo-100 text-indigo-700 cursor-not-allowed'
                    : 'bg-gray-100 hover:bg-indigo-50 text-gray-700'
                }`}
              >
                {modelName}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Comparison Table */}
      {selectedModels.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Square2StackIcon className="h-16 w-16 mx-auto mb-4 opacity-30" />
          <p className="text-lg font-medium">No models selected</p>
          <p className="text-sm">Select 2-4 models above to start comparing</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="text-left p-3 bg-gray-50 sticky left-0 z-10 min-w-[200px]">
                  Feature
                </th>
                {selectedModels.map((modelName) => (
                  <th key={modelName} className="p-3 bg-gray-50 min-w-[200px]">
                    <div className="flex items-start justify-between">
                      <div className="font-semibold text-gray-800 text-left">{modelName}</div>
                      <button
                        onClick={() => toggleModel(modelName)}
                        className="text-gray-400 hover:text-red-600 ml-2"
                      >
                        <XMarkIcon className="h-5 w-5" />
                      </button>
                    </div>
                    <div className="text-xs text-gray-500 font-normal mt-1">
                      {LLM_DATABASE[modelName].parameters}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {/* Description */}
              <tr className="border-b border-gray-200">
                <td className="p-3 font-medium text-gray-700 bg-gray-50 sticky left-0">
                  Description
                </td>
                {selectedModels.map((modelName) => (
                  <td key={modelName} className="p-3 text-sm text-gray-600">
                    {LLM_DATABASE[modelName].description}
                  </td>
                ))}
              </tr>

              {/* RAM Requirements */}
              <tr className="border-b border-gray-200 bg-blue-50">
                <td className="p-3 font-medium text-gray-700 bg-blue-100 sticky left-0">
                  RAM Required
                </td>
                {selectedModels.map((modelName) => {
                  const model = LLM_DATABASE[modelName];
                  const canRun = systemInfo.totalRamGB >= model.min_ram_gb;
                  return (
                    <td key={modelName} className="p-3">
                      <div className="flex items-center">
                        {canRun ? (
                          <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                        ) : (
                          <XCircleIcon className="h-5 w-5 text-red-500 mr-2" />
                        )}
                        <div className="text-sm">
                          <div className="font-semibold">
                            {model.min_ram_gb} GB min
                          </div>
                          <div className="text-gray-500">
                            {model.recommended_ram_gb} GB recommended
                          </div>
                        </div>
                      </div>
                    </td>
                  );
                })}
              </tr>

              {/* VRAM Requirements */}
              <tr className="border-b border-gray-200">
                <td className="p-3 font-medium text-gray-700 bg-gray-50 sticky left-0">
                  VRAM Required
                </td>
                {selectedModels.map((modelName) => {
                  const model = LLM_DATABASE[modelName];
                  const hasGPU = systemInfo.gpus && systemInfo.gpus.length > 0;
                  return (
                    <td key={modelName} className="p-3 text-sm">
                      <div>
                        <div className="font-semibold text-gray-700">
                          {model.min_vram_gb} GB min
                        </div>
                        <div className="text-gray-500">
                          {model.recommended_vram_gb} GB recommended
                        </div>
                        {!hasGPU && (
                          <div className="text-xs text-orange-600 mt-1">
                            CPU-only mode available
                          </div>
                        )}
                      </div>
                    </td>
                  );
                })}
              </tr>

              {/* Performance Estimate */}
              <tr className="border-b border-gray-200 bg-green-50">
                <td className="p-3 font-medium text-gray-700 bg-green-100 sticky left-0">
                  Estimated Speed
                </td>
                {selectedModels.map((modelName) => {
                  const model = LLM_DATABASE[modelName];
                  const perf = performanceEstimator.estimatePerformance(model);
                  return (
                    <td key={modelName} className="p-3">
                      <div className="text-sm">
                        <div className="font-bold text-lg text-green-700">
                          {perf.tokensPerSecond} t/s
                        </div>
                        <div className={`text-xs font-semibold ${
                          perf.speedRating === 'Very Fast' ? 'text-green-600' :
                          perf.speedRating === 'Fast' ? 'text-blue-600' :
                          perf.speedRating === 'Moderate' ? 'text-yellow-600' :
                          'text-red-600'
                        }`}>
                          {perf.speedRating}
                        </div>
                        <div className="text-gray-500 mt-1">
                          Load: {perf.loadTimeSeconds}s
                        </div>
                      </div>
                    </td>
                  );
                })}
              </tr>

              {/* CPU Only */}
              <tr className="border-b border-gray-200">
                <td className="p-3 font-medium text-gray-700 bg-gray-50 sticky left-0">
                  CPU Compatible
                </td>
                {selectedModels.map((modelName) => {
                  const model = LLM_DATABASE[modelName];
                  return (
                    <td key={modelName} className="p-3">
                      {model.cpu_only ? (
                        <span className="inline-flex items-center text-green-600 text-sm font-semibold">
                          <CheckCircleIcon className="h-5 w-5 mr-1" />
                          Yes
                        </span>
                      ) : (
                        <span className="inline-flex items-center text-red-600 text-sm font-semibold">
                          <XCircleIcon className="h-5 w-5 mr-1" />
                          GPU Required
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>

              {/* Ollama Support */}
              <tr className="border-b border-gray-200 bg-purple-50">
                <td className="p-3 font-medium text-gray-700 bg-purple-100 sticky left-0">
                  Ollama Command
                </td>
                {selectedModels.map((modelName) => {
                  const model = LLM_DATABASE[modelName];
                  return (
                    <td key={modelName} className="p-3">
                      {model.install_methods.ollama ? (
                        <code className="text-xs bg-gray-800 text-green-400 px-2 py-1 rounded block">
                          {model.install_methods.ollama.command}
                        </code>
                      ) : (
                        <span className="text-sm text-gray-500">Not available</span>
                      )}
                    </td>
                  );
                })}
              </tr>

              {/* Domain */}
              {selectedModels.some(m => LLM_DATABASE[m].domain) && (
                <tr className="border-b border-gray-200">
                  <td className="p-3 font-medium text-gray-700 bg-gray-50 sticky left-0">
                    Specialization
                  </td>
                  {selectedModels.map((modelName) => {
                    const model = LLM_DATABASE[modelName];
                    return (
                      <td key={modelName} className="p-3">
                        {model.domain ? (
                          <span className="inline-block bg-indigo-100 text-indigo-700 text-xs font-semibold px-2 py-1 rounded">
                            {model.domain}
                          </span>
                        ) : (
                          <span className="text-sm text-gray-500">General Purpose</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default ModelComparison;
