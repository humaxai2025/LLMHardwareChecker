'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChartBarIcon, CpuChipIcon, ScaleIcon } from '@heroicons/react/24/outline';

interface QuantizationOption {
  name: string;
  bitsPerWeight: number;
  sizeMultiplier: number;
  qualityScore: number;
  speedMultiplier: number;
  description: string;
  recommended: string;
}

const quantizationOptions: QuantizationOption[] = [
  {
    name: 'Q2_K',
    bitsPerWeight: 2.5,
    sizeMultiplier: 0.35,
    qualityScore: 70,
    speedMultiplier: 1.4,
    description: 'Smallest size, lowest quality',
    recommended: 'Very limited VRAM (<6GB), basic tasks only',
  },
  {
    name: 'Q3_K_M',
    bitsPerWeight: 3.5,
    sizeMultiplier: 0.5,
    qualityScore: 80,
    speedMultiplier: 1.25,
    description: 'Small size, acceptable quality',
    recommended: 'Limited VRAM (6-8GB), general use',
  },
  {
    name: 'Q4_K_M',
    bitsPerWeight: 4.5,
    sizeMultiplier: 0.65,
    qualityScore: 90,
    speedMultiplier: 1.15,
    description: 'Balanced size and quality (Most Popular)',
    recommended: 'Moderate VRAM (8-12GB), best balance',
  },
  {
    name: 'Q5_K_M',
    bitsPerWeight: 5.5,
    sizeMultiplier: 0.8,
    qualityScore: 95,
    speedMultiplier: 1.05,
    description: 'High quality, larger size',
    recommended: 'Good VRAM (12-16GB), quality-focused',
  },
  {
    name: 'Q8_0',
    bitsPerWeight: 8,
    sizeMultiplier: 0.95,
    qualityScore: 98,
    speedMultiplier: 0.95,
    description: 'Near-original quality',
    recommended: 'High VRAM (16GB+), maximum quality',
  },
  {
    name: 'FP16',
    bitsPerWeight: 16,
    sizeMultiplier: 1.0,
    qualityScore: 100,
    speedMultiplier: 0.85,
    description: 'Full precision, largest size',
    recommended: 'Abundant VRAM (24GB+), research/benchmark',
  },
];

interface QuantizationCalculatorProps {
  modelSizeGB: number;
  availableMemoryGB: number;
  onSelect?: (quantization: string) => void;
}

const QuantizationCalculator: React.FC<QuantizationCalculatorProps> = ({
  modelSizeGB,
  availableMemoryGB,
  onSelect,
}) => {
  const [selectedQuant, setSelectedQuant] = useState<string>('Q4_K_M');

  const handleSelect = (quantName: string) => {
    setSelectedQuant(quantName);
    if (onSelect) {
      onSelect(quantName);
    }
  };

  const calculateFits = (option: QuantizationOption): boolean => {
    const quantizedSize = modelSizeGB * option.sizeMultiplier;
    return quantizedSize <= availableMemoryGB * 0.9; // Leave 10% headroom
  };

  const calculateSize = (option: QuantizationOption): number => {
    return modelSizeGB * option.sizeMultiplier;
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
      <div className="flex items-center mb-6">
        <ScaleIcon className="h-8 w-8 text-purple-500 mr-3" />
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Quantization Calculator</h2>
          <p className="text-sm text-gray-600">
            Choose the best quality vs size tradeoff for your hardware
          </p>
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="text-sm text-blue-600 font-medium mb-1">Base Model Size</div>
          <div className="text-2xl font-bold text-blue-700">{modelSizeGB.toFixed(1)} GB</div>
        </div>
        <div className="bg-green-50 rounded-lg p-4">
          <div className="text-sm text-green-600 font-medium mb-1">Available Memory</div>
          <div className="text-2xl font-bold text-green-700">{availableMemoryGB} GB</div>
        </div>
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="text-sm text-purple-600 font-medium mb-1">Selected Size</div>
          <div className="text-2xl font-bold text-purple-700">
            {calculateSize(quantizationOptions.find(q => q.name === selectedQuant)!).toFixed(1)} GB
          </div>
        </div>
      </div>

      {/* Options */}
      <div className="space-y-3">
        {quantizationOptions.map((option) => {
          const fits = calculateFits(option);
          const size = calculateSize(option);
          const isSelected = selectedQuant === option.name;

          return (
            <motion.div
              key={option.name}
              whileHover={{ scale: fits ? 1.02 : 1 }}
              className={`relative border-2 rounded-lg p-4 transition-all cursor-pointer ${
                isSelected
                  ? 'border-purple-500 bg-purple-50'
                  : fits
                  ? 'border-gray-200 hover:border-purple-300 bg-white'
                  : 'border-gray-200 bg-gray-50 opacity-60 cursor-not-allowed'
              }`}
              onClick={() => fits && handleSelect(option.name)}
            >
              {!fits && (
                <div className="absolute top-2 right-2 text-xs bg-red-500 text-white px-2 py-1 rounded">
                  Won't Fit
                </div>
              )}

              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="text-lg font-bold text-gray-800">
                    {option.name}
                    {option.name === 'Q4_K_M' && (
                      <span className="ml-2 text-xs bg-blue-500 text-white px-2 py-1 rounded">
                        Recommended
                      </span>
                    )}
                  </h3>
                  <p className="text-sm text-gray-600">{option.description}</p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-gray-800">{size.toFixed(1)} GB</div>
                  <div className="text-xs text-gray-500">
                    {((size / availableMemoryGB) * 100).toFixed(0)}% of memory
                  </div>
                </div>
              </div>

              {/* Progress Bars */}
              <div className="grid grid-cols-3 gap-4 mb-2">
                {/* Quality */}
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-gray-600">Quality</span>
                    <span className="text-xs font-semibold text-gray-700">{option.qualityScore}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full transition-all"
                      style={{ width: `${option.qualityScore}%` }}
                    ></div>
                  </div>
                </div>

                {/* Speed */}
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-gray-600">Speed</span>
                    <span className="text-xs font-semibold text-gray-700">
                      {(option.speedMultiplier * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all"
                      style={{ width: `${Math.min(option.speedMultiplier * 100, 100)}%` }}
                    ></div>
                  </div>
                </div>

                {/* Size Efficiency */}
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-gray-600">Compression</span>
                    <span className="text-xs font-semibold text-gray-700">
                      {((1 - option.sizeMultiplier) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-purple-500 h-2 rounded-full transition-all"
                      style={{ width: `${(1 - option.sizeMultiplier) * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Recommendation */}
              <div className="text-xs text-gray-600 bg-gray-100 rounded px-3 py-2 mt-2">
                <strong>Best for:</strong> {option.recommended}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <h4 className="font-semibold text-blue-900 mb-2">Understanding Quantization</h4>
        <div className="text-sm text-blue-800 space-y-1">
          <p>
            <strong>Quantization</strong> reduces model size by using fewer bits per weight. Lower
            quantization = smaller files but slightly reduced quality.
          </p>
          <p>
            <strong>Q4_K_M</strong> is the sweet spot for most users, offering 90% quality at 65% of
            the original size.
          </p>
          <p>
            <strong>Tip:</strong> Start with Q4_K_M. If you have VRAM to spare, try Q5_K_M or Q8_0 for
            better quality.
          </p>
        </div>
      </div>
    </div>
  );
};

export default QuantizationCalculator;
