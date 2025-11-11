'use client';

import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  ChevronDownIcon,
  ChevronUpIcon,
  ClipboardDocumentIcon,
  CheckCircleIcon,
  XCircleIcon,
  ChartBarIcon,
  ArrowsUpDownIcon,
  CodeBracketIcon,
} from '@heroicons/react/24/outline';
import { Recommendations, ModelRecommendation } from '../lib/llmDatabase';
import { SystemInfo } from '../lib/systemAnalyzer';
import { PerformanceEstimator } from '../lib/performanceEstimator';
import ExportConfigurations from './ExportConfigurations';

interface ModelRecommendationsProps {
  recommendations: Recommendations;
  systemInfo: SystemInfo;
}

type SortKey = 'name' | 'parameters' | 'ram' | 'vram' | 'speed';
type SortOrder = 'asc' | 'desc';
type CategoryTab = 'excellent' | 'good' | 'basic' | 'not_suitable';

const ModelRecommendations: React.FC<ModelRecommendationsProps> = ({
  recommendations,
  systemInfo,
}) => {
  const [activeTab, setActiveTab] = useState<CategoryTab>('excellent');
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('parameters');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');
  const [showExportFor, setShowExportFor] = useState<string | null>(null);

  const performanceEstimator = new PerformanceEstimator(systemInfo);

  const parseParameterSize = (params: string): number => {
    const match = params.match(/(\d+\.?\d*)/);
    if (!match) return 0;
    const num = parseFloat(match[1]);
    if (params.includes('B')) return num;
    if (params.includes('M')) return num / 1000;
    return num;
  };

  const sortModels = (models: ModelRecommendation[]) => {
    return [...models].sort((a, b) => {
      let comparison = 0;

      switch (sortKey) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'parameters':
          comparison = parseParameterSize(a.specs.parameters) - parseParameterSize(b.specs.parameters);
          break;
        case 'ram':
          comparison = a.specs.min_ram_gb - b.specs.min_ram_gb;
          break;
        case 'vram':
          comparison = a.specs.min_vram_gb - b.specs.min_vram_gb;
          break;
        case 'speed':
          const perfA = performanceEstimator.estimatePerformance(a.specs);
          const perfB = performanceEstimator.estimatePerformance(b.specs);
          comparison = perfB.tokensPerSecond - perfA.tokensPerSecond;
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });
  };

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortOrder('asc');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      toast.success('Command copied!');
    });
  };

  const activeModels = useMemo(() => {
    return sortModels(recommendations[activeTab]);
  }, [activeTab, recommendations, sortKey, sortOrder]);

  const tabs = [
    { key: 'excellent' as CategoryTab, label: 'Excellent', count: recommendations.excellent.length, color: 'green' },
    { key: 'good' as CategoryTab, label: 'Good', count: recommendations.good.length, color: 'blue' },
    { key: 'basic' as CategoryTab, label: 'Basic', count: recommendations.basic.length, color: 'amber' },
    { key: 'not_suitable' as CategoryTab, label: 'Not Suitable', count: recommendations.not_suitable.length, color: 'red' },
  ];

  const getTabColorClasses = (color: string, isActive: boolean) => {
    const colors = {
      green: isActive ? 'bg-green-600 text-white' : 'text-green-700 hover:bg-green-50',
      blue: isActive ? 'bg-blue-600 text-white' : 'text-blue-700 hover:bg-blue-50',
      amber: isActive ? 'bg-amber-600 text-white' : 'text-amber-700 hover:bg-amber-50',
      red: isActive ? 'bg-red-600 text-white' : 'text-red-700 hover:bg-red-50',
    };
    return colors[color as keyof typeof colors];
  };

  const SortButton: React.FC<{ column: SortKey; label: string }> = ({ column, label }) => (
    <button
      onClick={() => handleSort(column)}
      className="flex items-center gap-1 hover:text-blue-600 transition-colors group"
      aria-label={`Sort by ${label}`}
    >
      {label}
      <ArrowsUpDownIcon className={`h-4 w-4 ${sortKey === column ? 'text-blue-600' : 'text-gray-400 group-hover:text-blue-600'}`} />
    </button>
  );

  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-5 border-b border-blue-800">
        <div className="flex items-center">
          <div className="p-2 bg-white bg-opacity-20 rounded-lg mr-4">
            <ChartBarIcon className="h-6 w-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">Model Recommendations</h2>
            <p className="text-blue-100 text-sm">Compatible models for your hardware</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 bg-gray-50">
        <div className="flex overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-6 py-3 font-medium text-sm whitespace-nowrap transition-colors border-b-2 ${
                activeTab === tab.key
                  ? 'border-blue-600 ' + getTabColorClasses(tab.color, true)
                  : 'border-transparent ' + getTabColorClasses(tab.color, false)
              }`}
              aria-label={`View ${tab.label} models`}
            >
              {tab.label}
              <span className={`ml-2 px-2 py-0.5 rounded-full text-xs font-semibold ${
                activeTab === tab.key ? 'bg-white bg-opacity-30' : 'bg-gray-200'
              }`}>
                {tab.count}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <SortButton column="name" label="Model" />
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <SortButton column="parameters" label="Size" />
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <SortButton column="ram" label="RAM" />
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <SortButton column="vram" label="VRAM" />
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <SortButton column="speed" label="Est. Speed" />
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                CPU Only
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {activeModels.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                  No models in this category
                </td>
              </tr>
            ) : (
              activeModels.map((model) => {
                const performance = performanceEstimator.estimatePerformance(model.specs);
                const isExpanded = expandedRow === model.name;

                return (
                  <React.Fragment key={model.name}>
                    <tr className="hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setExpandedRow(isExpanded ? null : model.name)}
                            className="text-gray-400 hover:text-blue-600"
                            aria-label={isExpanded ? 'Collapse details' : 'Expand details'}
                          >
                            {isExpanded ? (
                              <ChevronUpIcon className="h-5 w-5" />
                            ) : (
                              <ChevronDownIcon className="h-5 w-5" />
                            )}
                          </button>
                          <div>
                            <div className="font-semibold text-gray-900">{model.name}</div>
                            {model.specs.domain && (
                              <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded">
                                {model.specs.domain}
                              </span>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center">
                        <span className="font-medium text-gray-900">{model.specs.parameters}</span>
                      </td>
                      <td className="px-4 py-4 text-center">
                        <div className="text-sm">
                          <div className="font-semibold text-gray-900">{model.specs.min_ram_gb} GB</div>
                          <div className="text-xs text-gray-500">({model.specs.recommended_ram_gb} GB rec.)</div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center">
                        <div className="text-sm">
                          <div className="font-semibold text-gray-900">{model.specs.min_vram_gb} GB</div>
                          <div className="text-xs text-gray-500">({model.specs.recommended_vram_gb} GB rec.)</div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center">
                        <div className="text-sm">
                          <div className="font-bold text-blue-600">{performance.tokensPerSecond} t/s</div>
                          <div className={`text-xs font-medium ${
                            performance.speedRating === 'Very Fast' ? 'text-green-600' :
                            performance.speedRating === 'Fast' ? 'text-blue-600' :
                            performance.speedRating === 'Moderate' ? 'text-amber-600' :
                            'text-red-600'
                          }`}>
                            {performance.speedRating}
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center">
                        {model.specs.cpu_only ? (
                          <CheckCircleIcon className="h-5 w-5 text-green-500 mx-auto" />
                        ) : (
                          <XCircleIcon className="h-5 w-5 text-red-500 mx-auto" />
                        )}
                      </td>
                      <td className="px-4 py-4 text-center">
                        <button
                          onClick={() => setExpandedRow(isExpanded ? null : model.name)}
                          className="text-blue-600 hover:text-blue-700 font-medium text-sm"
                        >
                          {isExpanded ? 'Hide' : 'Details'}
                        </button>
                      </td>
                    </tr>

                    {/* Expanded Row */}
                    {isExpanded && (
                      <tr>
                        <td colSpan={7} className="px-4 py-4 bg-gray-50">
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="space-y-4"
                          >
                            {/* Description */}
                            <div>
                              <h4 className="font-semibold text-gray-900 mb-2">Description</h4>
                              <p className="text-sm text-gray-600">{model.specs.description}</p>
                            </div>

                            {/* Performance Details */}
                            <div>
                              <h4 className="font-semibold text-gray-900 mb-2">Performance Estimate</h4>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                <div className="bg-white rounded-lg p-3 border border-gray-200">
                                  <div className="text-xs text-gray-500 mb-1">Load Time</div>
                                  <div className="font-semibold text-gray-900">{performance.loadTimeSeconds}s</div>
                                </div>
                                <div className="bg-white rounded-lg p-3 border border-gray-200">
                                  <div className="text-xs text-gray-500 mb-1">Memory Usage</div>
                                  <div className="font-semibold text-gray-900">{performance.memoryUsagePercent}%</div>
                                </div>
                                <div className="bg-white rounded-lg p-3 border border-gray-200">
                                  <div className="text-xs text-gray-500 mb-1">GPU Usage</div>
                                  <div className="font-semibold text-gray-900">{performance.gpuUtilizationPercent}%</div>
                                </div>
                                <div className="bg-white rounded-lg p-3 border border-gray-200">
                                  <div className="text-xs text-gray-500 mb-1">Power Draw</div>
                                  <div className="font-semibold text-gray-900">{performance.powerConsumptionWatts}W</div>
                                </div>
                              </div>
                            </div>

                            {/* Installation Commands */}
                            {model.specs.install_methods.ollama && (
                              <div>
                                <h4 className="font-semibold text-gray-900 mb-2">Quick Install (Ollama)</h4>
                                <div className="flex items-center gap-2 bg-gray-900 rounded-lg p-3">
                                  <code className="flex-1 text-green-400 text-sm font-mono">
                                    {model.specs.install_methods.ollama.command}
                                  </code>
                                  <button
                                    onClick={() => copyToClipboard(model.specs.install_methods.ollama!.command!)}
                                    className="p-2 hover:bg-gray-800 rounded transition-colors"
                                    aria-label="Copy command"
                                  >
                                    <ClipboardDocumentIcon className="h-5 w-5 text-gray-400" />
                                  </button>
                                </div>
                              </div>
                            )}

                            {/* Compatibility Notes */}
                            {model.compatibility.notes.length > 0 && (
                              <div>
                                <h4 className="font-semibold text-gray-900 mb-2">Notes</h4>
                                <ul className="space-y-1">
                                  {model.compatibility.notes.map((note, idx) => (
                                    <li key={idx} className="text-sm text-gray-600 flex items-start gap-2">
                                      <span className="text-blue-600 mt-0.5">•</span>
                                      <span>{note}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Export Configuration Button */}
                            <div className="mt-4">
                              <button
                                onClick={() => setShowExportFor(showExportFor === model.name ? null : model.name)}
                                className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 transition-colors"
                              >
                                <CodeBracketIcon className="h-5 w-5" />
                                {showExportFor === model.name ? 'Hide' : 'Export'} Configuration
                              </button>
                            </div>

                            {/* Export Configuration Panel */}
                            {showExportFor === model.name && (
                              <div className="mt-4">
                                <ExportConfigurations model={model} systemInfo={systemInfo} />
                              </div>
                            )}
                          </motion.div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Footer Summary */}
      <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <span>Showing {activeModels.length} models in {tabs.find(t => t.key === activeTab)?.label} category</span>
          <span>Total compatible: {recommendations.excellent.length + recommendations.good.length + recommendations.basic.length} models</span>
        </div>
      </div>
    </div>
  );
};

export default ModelRecommendations;
