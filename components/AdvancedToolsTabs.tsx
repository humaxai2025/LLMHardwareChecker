'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  RocketLaunchIcon,
  BeakerIcon,
  ArrowsRightLeftIcon,
  FolderIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import { SystemInfo } from '../lib/systemAnalyzer';
import UpgradeAdvisor from './UpgradeAdvisor';
import ModelComparison from './ModelComparison';
import QuantizationCalculator from './QuantizationCalculator';
import ProfileManager from './ProfileManager';
import SystemMonitor from './SystemMonitor';

interface AdvancedToolsTabsProps {
  systemInfo: SystemInfo;
  currentCompatibleCount: number;
  onSelectProfile: (systemInfo: SystemInfo) => void;
}

type TabKey = 'monitor' | 'upgrades' | 'comparison' | 'quantization' | 'profiles';

const AdvancedToolsTabs: React.FC<AdvancedToolsTabsProps> = ({
  systemInfo,
  currentCompatibleCount,
  onSelectProfile,
}) => {
  const [activeTab, setActiveTab] = useState<TabKey>('monitor');

  const tabs = [
    {
      key: 'monitor' as TabKey,
      label: 'LLM Services',
      icon: <ChartBarIcon className="h-5 w-5" />,
      color: 'purple',
    },
    {
      key: 'upgrades' as TabKey,
      label: 'Hardware Upgrades',
      icon: <RocketLaunchIcon className="h-5 w-5" />,
      color: 'orange',
    },
    {
      key: 'comparison' as TabKey,
      label: 'Model Comparison',
      icon: <ArrowsRightLeftIcon className="h-5 w-5" />,
      color: 'indigo',
    },
    {
      key: 'quantization' as TabKey,
      label: 'Quantization Calculator',
      icon: <BeakerIcon className="h-5 w-5" />,
      color: 'green',
    },
    {
      key: 'profiles' as TabKey,
      label: 'Hardware Profiles',
      icon: <FolderIcon className="h-5 w-5" />,
      color: 'blue',
    },
  ];

  const getTabColorClasses = (color: string, isActive: boolean) => {
    const colors = {
      orange: isActive
        ? 'bg-orange-600 text-white border-orange-600'
        : 'text-orange-700 hover:bg-orange-50 border-transparent',
      purple: isActive
        ? 'bg-purple-600 text-white border-purple-600'
        : 'text-purple-700 hover:bg-purple-50 border-transparent',
      indigo: isActive
        ? 'bg-indigo-600 text-white border-indigo-600'
        : 'text-indigo-700 hover:bg-indigo-50 border-transparent',
      green: isActive
        ? 'bg-green-600 text-white border-green-600'
        : 'text-green-700 hover:bg-green-50 border-transparent',
      blue: isActive
        ? 'bg-blue-600 text-white border-blue-600'
        : 'text-blue-700 hover:bg-blue-50 border-transparent',
    };
    return colors[color as keyof typeof colors];
  };

  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4 border-b border-blue-800">
        <h2 className="text-2xl font-bold text-white">Advanced Tools</h2>
        <p className="text-blue-100 text-sm">Additional features to optimize your LLM setup</p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 bg-gray-50">
        <div className="flex overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 px-6 py-3 font-medium text-sm whitespace-nowrap transition-all border-b-2 ${getTabColorClasses(
                tab.color,
                activeTab === tab.key
              )}`}
              aria-label={`View ${tab.label}`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="p-6">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {activeTab === 'monitor' && (
            <SystemMonitor
              totalRamGB={systemInfo.totalRamGB}
              gpuCount={systemInfo.gpus?.length || 0}
            />
          )}

          {activeTab === 'upgrades' && (
            <UpgradeAdvisor
              systemInfo={systemInfo}
              currentCompatibleCount={currentCompatibleCount}
            />
          )}

          {activeTab === 'comparison' && <ModelComparison systemInfo={systemInfo} />}

          {activeTab === 'quantization' && (
            <QuantizationCalculator
              modelSizeGB={14}
              availableMemoryGB={systemInfo.totalRamGB}
            />
          )}

          {activeTab === 'profiles' && (
            <ProfileManager
              onSelectProfile={onSelectProfile}
              currentSystemInfo={systemInfo}
            />
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default AdvancedToolsTabs;
