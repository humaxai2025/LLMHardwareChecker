import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ComputerDesktopIcon, 
  CpuChipIcon, 
  CircleStackIcon,
  CheckIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { SystemInfo } from '../lib/systemAnalyzer';

interface ManualHardwareInputProps {
  onComplete: (systemInfo: SystemInfo) => void;
  browserDetected: Partial<SystemInfo>;
}

const ManualHardwareInput: React.FC<ManualHardwareInputProps> = ({
  onComplete,
  browserDetected
}) => {
  // Extract GPU info from browser detection
  const detectedGpu = browserDetected.gpus && browserDetected.gpus.length > 0
    ? browserDetected.gpus[0]
    : null;

  const [manualInput, setManualInput] = useState({
    os: browserDetected.os || 'Windows',
    processor: '',
    cpuCores: browserDetected.cpuCores || 8,
    totalRamGB: 16,
    gpuName: detectedGpu?.name || '',
    gpuVramGB: 8,
  });

  const [currentStep, setCurrentStep] = useState(0);

  const handleSubmit = () => {
    const systemInfo: SystemInfo = {
      os: manualInput.os,
      architecture: browserDetected.architecture || 'x64',
      processor: manualInput.processor || 'Unknown Processor',
      cpuCores: manualInput.cpuCores,
      totalRamGB: manualInput.totalRamGB,
      availableRamGB: manualInput.totalRamGB * 0.8,
      totalStorageGB: 500,
      freeStorageGB: 250,
      gpus: manualInput.gpuName ? [{
        name: manualInput.gpuName,
        vramGB: manualInput.gpuVramGB,
        type: manualInput.gpuName.toLowerCase().includes('nvidia') ? 'NVIDIA' : 
              manualInput.gpuName.toLowerCase().includes('amd') ? 'AMD' : 
              manualInput.gpuName.toLowerCase().includes('intel') ? 'Intel' : 'Unknown'
      }] : [],
      userAgent: browserDetected.userAgent || navigator.userAgent,
      screenResolution: browserDetected.screenResolution || `${screen.width}x${screen.height}`,
      colorDepth: browserDetected.colorDepth || screen.colorDepth,
      language: browserDetected.language || navigator.language,
      timezone: browserDetected.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone
    };

    onComplete(systemInfo);
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
          <h2 className="text-2xl font-bold text-white">
            Hardware Specification Input
          </h2>
          <p className="text-blue-100">
            Help us analyze your system accurately
          </p>
        </div>

        <div className="p-6">
          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-2">
              {[0, 1, 2].map((step, index) => (
                <div key={index} className="flex items-center">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    index <= currentStep 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-600'
                  }`}>
                    {index < currentStep ? <CheckIcon className="h-5 w-5" /> : index + 1}
                  </div>
                  {index < 2 && (
                    <div className={`w-16 h-1 mx-2 ${
                      index < currentStep ? 'bg-blue-600' : 'bg-gray-200'
                    }`} />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Step Content */}
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            {/* Step 0: OS & CPU */}
            {currentStep === 0 && (
              <div>
                <div className="flex items-center mb-6">
                  <ComputerDesktopIcon className="h-8 w-8 text-blue-600 mr-3" />
                  <h3 className="text-xl font-semibold text-gray-900">
                    Operating System & CPU
                  </h3>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Operating System
                    </label>
                    <select
                      value={manualInput.os}
                      onChange={(e) => setManualInput(prev => ({ ...prev, os: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="Windows">Windows</option>
                      <option value="macOS">macOS</option>
                      <option value="Linux">Linux</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Processor (e.g., AMD Ryzen 7 5800X, Intel Core i7-12700K)
                    </label>
                    <input
                      type="text"
                      value={manualInput.processor}
                      onChange={(e) => setManualInput(prev => ({ ...prev, processor: e.target.value }))}
                      placeholder="Enter your processor model"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      CPU Cores/Threads
                    </label>
                    <input
                      type="number"
                      value={manualInput.cpuCores}
                      onChange={(e) => setManualInput(prev => ({ ...prev, cpuCores: parseInt(e.target.value) || 8 }))}
                      min="1"
                      max="128"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Browser detected: {browserDetected.cpuCores} threads
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Step 1: RAM */}
            {currentStep === 1 && (
              <div>
                <div className="flex items-center mb-6">
                  <CircleStackIcon className="h-8 w-8 text-blue-600 mr-3" />
                  <h3 className="text-xl font-semibold text-gray-900">
                    Memory (RAM)
                  </h3>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Total RAM (GB)
                    </label>
                    <select
                      value={manualInput.totalRamGB}
                      onChange={(e) => setManualInput(prev => ({ ...prev, totalRamGB: parseInt(e.target.value) }))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value={4}>4 GB</option>
                      <option value={8}>8 GB</option>
                      <option value={16}>16 GB</option>
                      <option value={32}>32 GB</option>
                      <option value={64}>64 GB</option>
                      <option value={128}>128 GB</option>
                    </select>
                  </div>
                  
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex">
                      <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400 mr-2 flex-shrink-0" />
                      <div>
                        <h4 className="text-sm font-medium text-yellow-800">Why Manual Input?</h4>
                        <p className="text-sm text-yellow-700 mt-1">
                          For security reasons, browsers cannot access your actual RAM amount. 
                          Please enter your system's actual RAM for accurate LLM recommendations.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 2: GPU */}
            {currentStep === 2 && (
              <div>
                <div className="flex items-center mb-6">
                  <CpuChipIcon className="h-8 w-8 text-blue-600 mr-3" />
                  <h3 className="text-xl font-semibold text-gray-900">
                    Graphics Card (GPU)
                  </h3>
                </div>

                <div className="space-y-4">
                  {browserDetected.gpus && browserDetected.gpus.length > 0 && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div>
                          <h4 className="text-sm font-medium text-green-800 mb-1">✓ GPU Detected</h4>
                          <div className="text-sm text-green-700 font-medium">
                            {browserDetected.gpus[0].name}
                          </div>
                          <p className="text-xs text-green-600 mt-1">
                            This GPU has been automatically populated. You can edit or specify VRAM below.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      GPU Model {!detectedGpu && '(Optional)'}
                    </label>
                    <input
                      type="text"
                      value={manualInput.gpuName}
                      onChange={(e) => setManualInput(prev => ({ ...prev, gpuName: e.target.value }))}
                      placeholder="e.g., NVIDIA RTX 4090, AMD RX 7900 XTX"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    {!manualInput.gpuName && !detectedGpu && (
                      <p className="text-xs text-gray-500 mt-1">
                        Leave blank if you don't have a dedicated GPU
                      </p>
                    )}
                  </div>

                  {manualInput.gpuName && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        GPU VRAM (GB)
                      </label>
                      <select
                        value={manualInput.gpuVramGB}
                        onChange={(e) => setManualInput(prev => ({ ...prev, gpuVramGB: parseInt(e.target.value) }))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value={2}>2 GB</option>
                        <option value={4}>4 GB</option>
                        <option value={6}>6 GB</option>
                        <option value={8}>8 GB</option>
                        <option value={10}>10 GB</option>
                        <option value={12}>12 GB</option>
                        <option value={16}>16 GB</option>
                        <option value={24}>24 GB</option>
                        <option value={48}>48 GB</option>
                      </select>
                      <p className="text-xs text-gray-500 mt-1">
                        Browsers cannot detect VRAM amount. Please select your GPU's VRAM.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </motion.div>

          {/* Navigation */}
          <div className="flex justify-between mt-8">
            <button
              onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
              disabled={currentStep === 0}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>

            {currentStep < 2 ? (
              <button
                onClick={() => setCurrentStep(currentStep + 1)}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700"
              >
                Next
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                className="px-6 py-2 text-sm font-medium text-white bg-green-600 border border-transparent rounded-md hover:bg-green-700"
              >
                <CheckIcon className="h-4 w-4 inline mr-2" />
                Analyze My System
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ManualHardwareInput;