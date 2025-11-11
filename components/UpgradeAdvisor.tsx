'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  RocketLaunchIcon,
  CpuChipIcon,
  CircleStackIcon,
  VideoCameraIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline';
import { SystemInfo } from '../lib/systemAnalyzer';
import { LLM_DATABASE } from '../lib/llmDatabase';

interface UpgradeRecommendation {
  type: 'RAM' | 'GPU' | 'CPU';
  currentValue: string;
  recommendedValue: string;
  cost: string;
  benefit: string;
  priority: 'High' | 'Medium' | 'Low';
  newModelsUnlocked: number;
  icon: React.ReactNode;
}

interface UpgradeAdvisorProps {
  systemInfo: SystemInfo;
  currentCompatibleCount: number;
}

const UpgradeAdvisor: React.FC<UpgradeAdvisorProps> = ({
  systemInfo,
  currentCompatibleCount,
}) => {
  const generateRecommendations = (): UpgradeRecommendation[] => {
    const recommendations: UpgradeRecommendation[] = [];
    const allModels = Object.entries(LLM_DATABASE);
    const totalModels = allModels.length;

    // RAM Upgrade
    if (systemInfo.totalRamGB < 32) {
      const targetRAM = systemInfo.totalRamGB < 16 ? 32 : 64;
      const newlyCompatible = allModels.filter(([_, model]) =>
        model.min_ram_gb <= targetRAM && model.min_ram_gb > systemInfo.totalRamGB
      ).length;

      recommendations.push({
        type: 'RAM',
        currentValue: `${systemInfo.totalRamGB} GB`,
        recommendedValue: `${targetRAM} GB`,
        cost: systemInfo.totalRamGB < 16 ? '$60-$120' : '$120-$250',
        benefit: `Run ${newlyCompatible} additional models including larger 13B-70B models`,
        priority: systemInfo.totalRamGB < 16 ? 'High' : 'Medium',
        newModelsUnlocked: newlyCompatible,
        icon: <CircleStackIcon className="h-6 w-6" />,
      });
    }

    // GPU Upgrade
    const hasGPU = systemInfo.gpus && systemInfo.gpus.length > 0;
    const gpuCount = hasGPU ? systemInfo.gpus!.length : 0;
    const currentVRAM = hasGPU && systemInfo.gpus![0].vramGB
      ? (typeof systemInfo.gpus![0].vramGB === 'string'
        ? parseInt(systemInfo.gpus![0].vramGB) || 0
        : systemInfo.gpus![0].vramGB)
      : 0;

    // Calculate total VRAM if multiple GPUs
    const totalVRAM = hasGPU
      ? systemInfo.gpus!.reduce((sum, gpu) => {
          const vram = typeof gpu.vramGB === 'string' ? parseInt(gpu.vramGB) || 0 : gpu.vramGB || 0;
          return sum + vram;
        }, 0)
      : 0;

    // Multi-GPU optimization recommendation
    if (gpuCount > 1 && totalVRAM > 0) {
      recommendations.push({
        type: 'GPU',
        currentValue: `${gpuCount} GPUs (${totalVRAM} GB total VRAM)`,
        recommendedValue: 'Multi-GPU optimization',
        cost: '$0 (configuration only)',
        benefit: `Use tensor parallelism to run larger models (up to ${Math.floor(totalVRAM * 0.8)}B parameters). Enable with --tensor-split in llama.cpp or multi_gpu: true in Ollama`,
        priority: 'High',
        newModelsUnlocked: allModels.filter(([_, model]) => {
          const paramSize = parseFloat(model.parameters.match(/(\d+\.?\d*)/)?.[1] || '0');
          return model.min_vram_gb > currentVRAM && model.min_vram_gb <= totalVRAM * 0.8;
        }).length,
        icon: <VideoCameraIcon className="h-6 w-6" />,
      });
    }

    // Single GPU upgrade
    if (!hasGPU || (gpuCount === 1 && currentVRAM < 12)) {
      const targetVRAM = !hasGPU ? 8 : currentVRAM < 8 ? 12 : 16;
      const gpuName = !hasGPU
        ? 'RTX 4060 Ti (16GB) or RX 7600 XT'
        : currentVRAM < 8
        ? 'RTX 4070 (12GB) or RX 7700 XT'
        : 'RTX 4070 Ti Super (16GB) or RX 7800 XT';

      const newlyCompatible = allModels.filter(([_, model]) =>
        model.min_vram_gb <= targetVRAM && model.min_vram_gb > currentVRAM
      ).length;

      const speedBoost = !hasGPU ? '5-10x faster' : '2-3x faster';

      recommendations.push({
        type: 'GPU',
        currentValue: hasGPU ? `${currentVRAM} GB VRAM` : 'No dedicated GPU',
        recommendedValue: `${gpuName}`,
        cost: !hasGPU ? '$400-$600' : currentVRAM < 8 ? '$500-$700' : '$700-$900',
        benefit: `${speedBoost} inference, unlock ${newlyCompatible} GPU-optimized models`,
        priority: !hasGPU ? 'High' : 'Medium',
        newModelsUnlocked: newlyCompatible,
        icon: <VideoCameraIcon className="h-6 w-6" />,
      });
    }

    // Second GPU recommendation for large models
    if (gpuCount === 1 && currentVRAM >= 8 && currentVRAM < 24) {
      const combinedVRAM = currentVRAM * 2;
      const newlyCompatible = allModels.filter(([_, model]) =>
        model.min_vram_gb <= combinedVRAM * 0.8 && model.min_vram_gb > currentVRAM
      ).length;

      recommendations.push({
        type: 'GPU',
        currentValue: `1x GPU (${currentVRAM} GB VRAM)`,
        recommendedValue: `Add second identical GPU for multi-GPU setup`,
        cost: `$${currentVRAM <= 8 ? '400-600' : '600-900'} (used recommended)`,
        benefit: `Run models up to ${Math.floor(combinedVRAM * 0.8)}B parameters with tensor parallelism. Unlock ${newlyCompatible} large models`,
        priority: 'Low',
        newModelsUnlocked: newlyCompatible,
        icon: <VideoCameraIcon className="h-6 w-6" />,
      });
    }

    // CPU Upgrade (lower priority)
    if (systemInfo.cpuCores < 8) {
      recommendations.push({
        type: 'CPU',
        currentValue: `${systemInfo.cpuCores} cores`,
        recommendedValue: '8+ cores (e.g., AMD Ryzen 7 or Intel i7)',
        cost: '$200-$400',
        benefit: '25-40% faster CPU inference, better multitasking',
        priority: 'Low',
        newModelsUnlocked: 0,
        icon: <CpuChipIcon className="h-6 w-6" />,
      });
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { High: 0, Medium: 1, Low: 2 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });
  };

  const recommendations = generateRecommendations();

  const calculateImpact = () => {
    const totalNewModels = recommendations.reduce((sum, rec) => sum + rec.newModelsUnlocked, 0);
    const potentialTotal = currentCompatibleCount + totalNewModels;
    return { totalNewModels, potentialTotal };
  };

  const { totalNewModels, potentialTotal } = calculateImpact();

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'High':
        return 'bg-red-100 text-red-700 border-red-300';
      case 'Medium':
        return 'bg-yellow-100 text-yellow-700 border-yellow-300';
      case 'Low':
        return 'bg-green-100 text-green-700 border-green-300';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-300';
    }
  };

  if (recommendations.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
        <div className="flex items-center mb-4">
          <RocketLaunchIcon className="h-8 w-8 text-green-500 mr-3" />
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Hardware Upgrade Advisor</h2>
            <p className="text-sm text-gray-600">Optimize your system for better LLM performance</p>
          </div>
        </div>

        <div className="text-center py-12">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-green-100 rounded-full mb-4">
            <RocketLaunchIcon className="h-10 w-10 text-green-600" />
          </div>
          <h3 className="text-xl font-bold text-gray-800 mb-2">Excellent Hardware!</h3>
          <p className="text-gray-600">
            Your system is well-equipped to run most LLM models. No immediate upgrades needed.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
      <div className="flex items-center mb-6">
        <RocketLaunchIcon className="h-8 w-8 text-orange-500 mr-3" />
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Hardware Upgrade Advisor</h2>
          <p className="text-sm text-gray-600">Suggested upgrades to unlock more models</p>
        </div>
      </div>

      {/* Impact Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
          <div className="text-sm text-blue-600 font-medium mb-1">Current Compatible</div>
          <div className="text-3xl font-bold text-blue-700">{currentCompatibleCount}</div>
          <div className="text-xs text-blue-600">models you can run now</div>
        </div>
        <div className="bg-green-50 rounded-lg p-4 border-2 border-green-200">
          <div className="text-sm text-green-600 font-medium mb-1">After Upgrades</div>
          <div className="text-3xl font-bold text-green-700">{potentialTotal}</div>
          <div className="text-xs text-green-600">total compatible models</div>
        </div>
        <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200">
          <div className="text-sm text-purple-600 font-medium mb-1">New Models</div>
          <div className="text-3xl font-bold text-purple-700">+{totalNewModels}</div>
          <div className="text-xs text-purple-600">additional models unlocked</div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="space-y-4">
        {recommendations.map((rec, index) => (
          <motion.div
            key={rec.type}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="border-2 border-gray-200 rounded-lg p-5 hover:border-orange-300 transition-all"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-start">
                <div className="p-2 bg-orange-100 rounded-lg mr-4">
                  {rec.icon}
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-xl font-bold text-gray-800">{rec.type} Upgrade</h3>
                    <span
                      className={`text-xs font-semibold px-2 py-1 rounded border ${getPriorityColor(
                        rec.priority
                      )}`}
                    >
                      {rec.priority} Priority
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 space-y-1">
                    <div>
                      <span className="font-medium">Current:</span> {rec.currentValue}
                    </div>
                    <div>
                      <span className="font-medium">Recommended:</span> {rec.recommendedValue}
                    </div>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-orange-600">{rec.cost}</div>
                <div className="text-xs text-gray-500">Estimated cost</div>
              </div>
            </div>

            <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
              <div className="flex items-start">
                <ArrowTrendingUpIcon className="h-5 w-5 text-orange-600 mr-2 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-gray-700">
                  <strong className="text-orange-700">Impact:</strong> {rec.benefit}
                </div>
              </div>
            </div>

            {rec.newModelsUnlocked > 0 && (
              <div className="mt-3 text-sm text-gray-600">
                <strong>Unlocks {rec.newModelsUnlocked} new models</strong> including more powerful
                options for coding, reasoning, and general tasks
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Tips */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <h4 className="font-semibold text-blue-900 mb-2">💡 Upgrade Tips</h4>
        <ul className="text-sm text-blue-800 space-y-2 list-disc list-inside">
          <li>
            <strong>Start with RAM:</strong> Most cost-effective upgrade for running larger models
          </li>
          <li>
            <strong>GPU is key for speed:</strong> 5-10x faster inference than CPU-only
          </li>
          <li>
            <strong>Watch for sales:</strong> GPU prices fluctuate, wait for deals on last-gen cards
          </li>
          <li>
            <strong>Used market:</strong> Consider used enterprise GPUs (e.g., A4000, P40) for better
            value
          </li>
          <li>
            <strong>Incremental upgrades:</strong> You don't need to buy everything at once
          </li>
        </ul>
      </div>
    </div>
  );
};

export default UpgradeAdvisor;
