'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast, Toaster } from 'react-hot-toast';
import { 
  ComputerDesktopIcon, 
  CpuChipIcon, 
  DocumentArrowDownIcon,
  PlayIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';

import { SystemInfo } from '../lib/systemAnalyzer';
import { LLMRecommender } from '../lib/llmRecommender';
import { Recommendations } from '../lib/llmDatabase';
import { analyzeClientSystem } from '../lib/clientAnalyzer';

// Components
import LoadingSpinner from '../components/LoadingSpinner';
import SystemSpecsCard from '../components/SystemSpecsCard';
import ModelRecommendations from '../components/ModelRecommendations';
import InstallationGuide from '../components/InstallationGuide';
import OptimizationTips from '../components/OptimizationTips';
import ReportDownload from '../components/ReportDownload';
import ErrorBoundary from '../components/ErrorBoundary';
import FeedbackButton from '../components/FeedbackButton';

interface AnalysisState {
  systemInfo: SystemInfo | null;
  recommendations: Recommendations | null;
  recommender: LLMRecommender | null;
  isLoading: boolean;
  isAnalyzing: boolean;
  error: string | null;
  analysisComplete: boolean;
}

export default function HomePage() {
  const [isMounted, setIsMounted] = useState(false);
  const [state, setState] = useState<AnalysisState>({
    systemInfo: null,
    recommendations: null,
    recommender: null,
    isLoading: false,
    isAnalyzing: false,
    error: null,
    analysisComplete: false,
  });

  // Ensure we only run client-side code after mount
  useEffect(() => {
    setIsMounted(true);
  }, []);

  const startAnalysis = async () => {
    // Multiple checks to ensure we're in browser
    if (typeof window === 'undefined' || typeof document === 'undefined' || typeof navigator === 'undefined') {
      setState(prev => ({ 
        ...prev, 
        error: 'Analysis must run in a browser environment. Please ensure JavaScript is enabled.',
        isLoading: false,
        isAnalyzing: false 
      }));
      return;
    }

    // Check if we have basic browser APIs
    if (!navigator.userAgent || !navigator.hardwareConcurrency) {
      setState(prev => ({ 
        ...prev, 
        error: 'Browser does not support hardware detection APIs.',
        isLoading: false,
        isAnalyzing: false 
      }));
      return;
    }

    setState(prev => ({ ...prev, isLoading: true, isAnalyzing: true, error: null }));
    
    try {
      toast.loading('Analyzing your browser client hardware...', { id: 'analysis' });
      
      // Add delay for better UX
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // This will now ONLY analyze the CLIENT's hardware using dynamic import
      console.log('🔍 Starting client-side hardware analysis...');
      console.log('User Agent:', navigator.userAgent);
      console.log('Platform:', navigator.platform);
      console.log('CPU Cores:', navigator.hardwareConcurrency);
      
      const systemInfo = await analyzeClientSystem();
      
      console.log('✅ Client hardware analysis complete:', systemInfo);
      
      toast.loading('Generating recommendations based on your hardware...', { id: 'analysis' });
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const recommender = new LLMRecommender(systemInfo);
      const recommendations = recommender.getRecommendations();
      
      setState({
        systemInfo,
        recommendations,
        recommender,
        isLoading: false,
        isAnalyzing: false,
        error: null,
        analysisComplete: true,
      });
      
      toast.success('Your hardware analysis is complete!', { id: 'analysis' });
      
    } catch (error) {
      console.error('❌ Client-side analysis failed:', error);
      setState(prev => ({
        ...prev,
        isLoading: false,
        isAnalyzing: false,
        error: `Failed to analyze your system: ${error instanceof Error ? error.message : 'Please ensure you\'re using a modern browser and JavaScript is enabled.'}`,
      }));
      
      toast.error('Analysis failed. Please ensure you\'re using a modern browser.', { id: 'analysis' });
    }
  };

  const restartAnalysis = () => {
    setState({
      systemInfo: null,
      recommendations: null,
      recommender: null,
      isLoading: false,
      isAnalyzing: false,
      error: null,
      analysisComplete: false,
    });
  };

  const getSuitableModelsCount = () => {
    if (!state.recommendations) return 0;
    return (
      state.recommendations.excellent.length +
      state.recommendations.good.length +
      state.recommendations.basic.length
    );
  };

  const getCapabilityLevel = () => {
    const defaultLevel = { label: 'Unknown', color: 'text-gray-600', bg: 'bg-gray-100' };
    
    if (!state.recommender) return defaultLevel;
    
    const level = state.recommender.getSystemCapabilityLevel();
    const levelMap = {
      low: { label: 'Entry Level', color: 'text-orange-600', bg: 'bg-orange-100' },
      medium: { label: 'Mid Range', color: 'text-blue-600', bg: 'bg-blue-100' },
      high: { label: 'High End', color: 'text-green-600', bg: 'bg-green-100' },
      premium: { label: 'Premium', color: 'text-purple-600', bg: 'bg-purple-100' }
    };
    return levelMap[level] || defaultLevel;
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1f2937',
              color: '#f9fafb',
            },
          }}
        />
        
        {/* Header */}
        <header className="relative overflow-hidden bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600">
          <div className="absolute inset-0 bg-black opacity-10"></div>
          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center"
            >
              <div className="flex justify-center mb-6">
                <div className="p-3 bg-white bg-opacity-20 rounded-full">
                  <CpuChipIcon className="h-12 w-12 text-white" />
                </div>
              </div>
              <h1 className="text-5xl font-bold text-white mb-6">
                LLM Hardware Compatibility Checker
              </h1>
              <p className="text-xl text-blue-100 max-w-3xl mx-auto mb-8 leading-relaxed">
                Discover which Large Language Models your system can run locally. 
                Get personalized recommendations with detailed installation instructions.
              </p>
              
              {!state.analysisComplete && !isMounted && (
                <div className="inline-flex items-center px-8 py-4 bg-gray-200 text-gray-600 font-semibold rounded-full">
                  <LoadingSpinner className="mr-3" />
                  Loading...
                </div>
              )}
              
              {!state.analysisComplete && isMounted && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={startAnalysis}
                  disabled={state.isLoading}
                  className="inline-flex items-center px-8 py-4 bg-white text-blue-600 font-semibold rounded-full shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {state.isLoading ? (
                    <>
                      <LoadingSpinner className="mr-3" />
                      Analyzing System...
                    </>
                  ) : (
                    <>
                      <PlayIcon className="h-5 w-5 mr-3" />
                      Start Hardware Analysis
                    </>
                  )}
                </motion.button>
              )}
            </motion.div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <AnimatePresence mode="wait">
            {state.error && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="mb-8 bg-red-50 border border-red-200 rounded-lg p-6"
              >
                <div className="flex">
                  <ExclamationTriangleIcon className="h-6 w-6 text-red-400 mr-3 flex-shrink-0" />
                  <div>
                    <h3 className="text-lg font-medium text-red-800 mb-2">Analysis Failed</h3>
                    <p className="text-red-700 mb-4">{state.error}</p>
                    <button
                      onClick={startAnalysis}
                      className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition-colors"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {state.isAnalyzing && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="text-center py-20"
              >
                <LoadingSpinner size="large" className="mx-auto mb-6" />
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                  Analyzing Your System
                </h2>
                <p className="text-gray-600 max-w-md mx-auto mb-4">
                  We're detecting your hardware specifications and checking compatibility 
                  with popular LLM models. This will take just a moment.
                </p>
                {isMounted && (
                  <div className="text-sm text-blue-600 bg-blue-50 rounded-lg p-3 max-w-lg mx-auto">
                    <div className="mb-2"><strong>🔍 Analyzing your browser client:</strong></div>
                    <div>Platform: {navigator.platform}</div>
                    <div>Browser: {navigator.userAgent.split(' ')[0]}</div>
                    <div>CPU Cores: {navigator.hardwareConcurrency}</div>
                    <div className="text-xs mt-2 text-blue-500">
                      ✅ This is YOUR device, not a server!
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {state.analysisComplete && state.systemInfo && state.recommendations && state.recommender && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ staggerChildren: 0.1 }}
                className="space-y-8"
              >
                {/* Results Summary */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden"
                >
                  <div className="bg-gradient-to-r from-green-500 to-blue-500 px-6 py-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <CheckCircleIcon className="h-8 w-8 text-white mr-3" />
                        <div>
                          <h2 className="text-2xl font-bold text-white">Analysis Complete!</h2>
                          <p className="text-green-100">Your system has been analyzed successfully</p>
                        </div>
                      </div>
                      <button
                        onClick={restartAnalysis}
                        className="bg-white bg-opacity-20 text-white px-4 py-2 rounded-lg hover:bg-opacity-30 transition-colors"
                      >
                        Run Again
                      </button>
                    </div>
                  </div>
                  
                  <div className="p-6">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-blue-600 mb-2">
                          {getSuitableModelsCount()}
                        </div>
                        <div className="text-gray-600">Compatible Models</div>
                      </div>
                      <div className="text-center">
                        <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getCapabilityLevel().bg} ${getCapabilityLevel().color} mb-2`}>
                          {getCapabilityLevel().label}
                        </div>
                        <div className="text-gray-600">System Capability</div>
                      </div>
                      <div className="text-center">
                        <div className="text-3xl font-bold text-purple-600 mb-2">
                          {state.systemInfo.totalRamGB} GB
                        </div>
                        <div className="text-gray-600">Total RAM</div>
                      </div>
                      <div className="text-center">
                        <div className="text-3xl font-bold text-indigo-600 mb-2">
                          {state.systemInfo.gpus?.length || 0}
                        </div>
                        <div className="text-gray-600">GPU(s) Detected</div>
                      </div>
                    </div>
                  </div>
                </motion.div>

                {/* System Specifications */}
                <SystemSpecsCard systemInfo={state.systemInfo} />

                {/* Model Recommendations */}
                <ModelRecommendations 
                  recommendations={state.recommendations}
                  systemInfo={state.systemInfo}
                />

                {/* Installation Guide */}
                <InstallationGuide recommender={state.recommender} />

                {/* Optimization Tips */}
                <OptimizationTips recommender={state.recommender} />

                {/* Report Download */}
                <ReportDownload 
                  systemInfo={state.systemInfo}
                  recommendations={state.recommendations}
                  recommender={state.recommender}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Info Section */}
          {!state.analysisComplete && !state.isAnalyzing && isMounted && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8"
            >
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
                <div className="flex items-center mb-4">
                  <ComputerDesktopIcon className="h-8 w-8 text-blue-500 mr-3" />
                  <h3 className="text-xl font-semibold text-gray-800">System Analysis</h3>
                </div>
                <p className="text-gray-600">
                  We detect your hardware specifications including CPU, RAM, GPU, and storage 
                  to determine compatibility with various LLM models.
                </p>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
                <div className="flex items-center mb-4">
                  <CpuChipIcon className="h-8 w-8 text-purple-500 mr-3" />
                  <h3 className="text-xl font-semibold text-gray-800">Smart Recommendations</h3>
                </div>
                <p className="text-gray-600">
                  Get personalized model recommendations based on your system's capabilities,
                  from lightweight 2B models to powerful 70B+ models.
                </p>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
                <div className="flex items-center mb-4">
                  <DocumentArrowDownIcon className="h-8 w-8 text-green-500 mr-3" />
                  <h3 className="text-xl font-semibold text-gray-800">Detailed Reports</h3>
                </div>
                <p className="text-gray-600">
                  Download comprehensive reports with installation instructions, 
                  optimization tips, and platform-specific setup guides.
                </p>
              </div>
            </motion.div>
          )}

          {/* Privacy Notice */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-16 bg-blue-50 border border-blue-200 rounded-lg p-6"
          >
            <div className="flex">
              <InformationCircleIcon className="h-6 w-6 text-blue-400 mr-3 flex-shrink-0" />
              <div>
                <h3 className="text-lg font-medium text-blue-800 mb-2">Privacy & Security</h3>
                <p className="text-blue-700 mb-2">
                  All hardware analysis is performed <strong>locally in your browser</strong>. No data is sent to external servers. 
                  Your system information remains completely private and secure.
                </p>
                <div className="text-sm text-blue-600 bg-blue-100 rounded-lg p-3 mt-3">
                  <strong>🔒 Client-Side Only:</strong> We analyze YOUR device's hardware, not our server's hardware. 
                  Analysis happens entirely in your browser using JavaScript APIs.
                </div>
              </div>
            </div>
          </motion.div>
        </main>

        {/* Floating Feedback Button */}
        {isMounted && <FeedbackButton />}
      </div>
    </ErrorBoundary>
  );
}