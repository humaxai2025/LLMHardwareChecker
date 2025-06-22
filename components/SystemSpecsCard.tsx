import React from 'react';
import { motion } from 'framer-motion';
import { 
  ComputerDesktopIcon,
  CpuChipIcon,
  CircleStackIcon,
  ServerStackIcon
} from '@heroicons/react/24/outline';
import { SystemInfo } from '../lib/systemAnalyzer';

interface SystemSpecsCardProps {
  systemInfo: SystemInfo;
}

const SystemSpecsCard: React.FC<SystemSpecsCardProps> = ({ systemInfo }) => {
  const specs = [
    {
      icon: ComputerDesktopIcon,
      label: 'Operating System',
      value: `${systemInfo.os} (${systemInfo.architecture})`,
      color: 'text-blue-500'
    },
    {
      icon: CpuChipIcon,
      label: 'Processor',
      value: systemInfo.processor,
      color: 'text-purple-500'
    },
    {
      icon: CpuChipIcon,
      label: 'CPU Cores',
      value: `${systemInfo.cpuCores} cores`,
      color: 'text-indigo-500'
    },
    {
      icon: CircleStackIcon,
      label: 'Memory (RAM)',
      value: `${systemInfo.totalRamGB} GB total (${systemInfo.availableRamGB} GB available)`,
      color: 'text-green-500'
    },
    {
      icon: ServerStackIcon,
      label: 'Storage',
      value: `${systemInfo.freeStorageGB} GB free / ${systemInfo.totalStorageGB} GB total`,
      color: 'text-orange-500'
    }
  ];

  // Add GPU specs if available
  if (systemInfo.gpus && systemInfo.gpus.length > 0) {
    systemInfo.gpus.forEach((gpu, index) => {
      specs.push({
        icon: CpuChipIcon,
        label: `GPU ${index + 1}`,
        value: `${gpu.name} (${typeof gpu.vramGB === 'number' ? gpu.vramGB + ' GB VRAM' : gpu.vramGB})`,
        color: 'text-red-500'
      });
    });
  } else {
    specs.push({
      icon: CpuChipIcon,
      label: 'GPU',
      value: 'None detected',
      color: 'text-gray-500'
    });
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden"
    >
      <div className="bg-gradient-to-r from-slate-800 to-slate-600 px-6 py-4">
        <div className="flex items-center">
          <ComputerDesktopIcon className="h-8 w-8 text-white mr-3" />
          <h2 className="text-2xl font-bold text-white">System Specifications</h2>
        </div>
      </div>
      
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {specs.map((spec, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:shadow-md transition-shadow"
            >
              <div className="flex items-start">
                <spec.icon className={`h-6 w-6 ${spec.color} mr-3 flex-shrink-0 mt-1`} />
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide mb-1">
                    {spec.label}
                  </h3>
                  <p className="text-gray-900 font-medium break-words">
                    {spec.value}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
        
        {/* Additional System Info */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Additional Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Screen Resolution:</span>
              <p className="font-medium">{systemInfo.screenResolution}</p>
            </div>
            <div>
              <span className="text-gray-500">Color Depth:</span>
              <p className="font-medium">{systemInfo.colorDepth}-bit</p>
            </div>
            <div>
              <span className="text-gray-500">Language:</span>
              <p className="font-medium">{systemInfo.language}</p>
            </div>
            <div>
              <span className="text-gray-500">Timezone:</span>
              <p className="font-medium">{systemInfo.timezone}</p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default SystemSpecsCard;