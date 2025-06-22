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

// Import unified types
import {
  SystemInfo,
  Recommendations,
  LLMRecommender as LLMRecommenderInterface,
  AnalysisState,
  SystemCapabilityLevel,
  CAPABILITY_LEVELS
} from '../types';

// Import actual implementation classes
import { LLMRecommender } from '../lib/llmRecommender';
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
import ManualHardwareInput from '../components/ManualHardwareInput';

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
    showManualInput: false,
    browserDetected: null,
  });

  // Ensure we only run client-side code after mount
  useEffect(() => {
    setIsMounted(true);
  }, []);

  const detectBrowserCapabilities = async (): Promise<Partial<SystemInfo> | null> => {
    if (typeof window === 'undefined' || typeof navigator === 'undefined') {
      return null;
    }

    try {
      const browserDetected: Partial<SystemInfo> = {
        os: navigator.platform.includes('Win') ? 'Windows' :
            navigator.platform.includes('Mac') ? 'macOS' :
            navigator.platform.includes('Linux') ? 'Linux' : 'Unknown',
        architecture: navigator.platform,
        cpuCores: navigator.hardwareConcurrency || 4,
        userAgent: navigator.userAgent,
        screenResolution: `${screen.width}x${screen.height}`,
        colorDepth: screen.colorDepth,
        language: navigator.language,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
      };

      // Try to detect GPU via WebGL
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') as WebGLRenderingContext | null;
        if (gl) {
          const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
          if (debugInfo) {
            const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) as string;
            browserDetected.gpus = [{
              name: renderer || 'Unknown GPU',
              vramGB: 'Unknown',
              type: 'Detected via WebGL'
            }];
          }
        }
      } catch (gpuError) {
        console.warn('GPU detection failed:', gpuError);
        browserDetected.gpus = [];
      }

      return browserDetected;
    } catch (error) {
      console.error('Browser detection failed:', error);
      return null;
    }
  };

  const startAutomaticAnalysis = async (): Promise<void> => {
    setState(prev => ({ ...prev, isLoading: true, isAnalyzing: true, error: null }));
    
    try {
      toast.loading('Detecting browser capabilities...', { id: 'analysis' });
      
      const browserDetected = await detectBrowserCapabilities();
      
      // Show limitation warning and manual input
      setState(prev => ({ 
        ...prev, 
        browserDetected,
        showManualInput: true,
        isLoading: false,
        isAnalyzing: false 
      }));
      
      toast.dismiss('analysis');
      
    } catch (error) {
      console.error('Browser detection failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to detect browser capabilities.';
      
      setState(prev => ({
        ...prev,
        isLoading: false,
        isAnalyzing: false,
        error: errorMessage,
      }));
      
      toast.error('Detection failed. Please try manual input.', { id: 'analysis' });
    }
  };

  const handleManualInput = async (systemInfo: SystemInfo): Promise<void> => {
    setState(prev => ({ ...prev, isLoading: true, isAnalyzing: true, error: null }));
    
    try {
      toast.loading('Generating recommendations for your system...', { id: 'analysis' });
      
      // Small delay for better UX
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Create recommender instance and get recommendations
      const recommender = new LLMRecommender(systemInfo);
      const recommendations = recommender.getRecommendations();
      
      setState(prev => ({
        ...prev,
        systemInfo,
        recommendations,
        recommender,
        isLoading: false,
        isAnalyzing: false,
        error: null,
        analysisComplete: true,
        showManualInput: false,
        browserDetected: null,
      }));
      
      toast.success('Analysis complete with your hardware specs!', { id: 'analysis' });
      
    } catch (error) {
      console.error('Analysis failed:', error);
      const errorMessage = error instanceof Error 
        ? `Failed to analyze system: ${error.message}`
        : 'Failed to analyze system: Unknown error';
      
      setState(prev => ({
        ...prev,
        isLoading: false,
        isAnalyzing: false,
        error: errorMessage,
      }));
      
      toast.error('Analysis failed. Please try again.', { id: 'analysis' });
    }
  };

  const restartAnalysis = (): void => {
    setState({
      systemInfo: null,
      recommendations: null,
      recommender: null,
      isLoading: false,
      isAnalyzing: false,
      error: null,
      analysisComplete: false,
      showManualInput: false,
      browserDetected: null,
    });
    
    // Clear any existing toasts
    toast.dismiss();
  };

  const getSuitableModelsCount = (): number => {
    if (!state.recommendations) return 0;
    return (
      state.recommendations.excellent.length +
      state.recommendations.good.length +
      state.recommendations.basic.length
    );
  };

  const getCapabilityLevel = (): { label: string; color: string; bg: string } => {
    const defaultLevel = { label: 'Unknown', color: 'text-gray-600', bg: 'bg-gray-100' };
    
    if (!state.recommender) return defaultLevel;
    
    try {
      const level: SystemCapabilityLevel = state.recommender.getSystemCapabilityLevel();
      const levelInfo = CAPABILITY_LEVELS[level];
      
      const colorMap = {
        orange: { color: 'text-orange-600', bg: 'bg-orange-100' },
        blue: { color: 'text-blue-600', bg: 'bg-blue-100' },
        green: { color: 'text-green-600', bg: 'bg-green-100' },
        purple: { color: 'text-purple-600', bg: 'bg-purple-100' }
      };
      
      const colors = colorMap[levelInfo.color as keyof typeof colorMap] || colorMap.blue;
      
      return {
        label: levelInfo.label,
        ...colors
      };
    } catch (error) {
      console.warn('Failed to get capability level:', error);
      return defaultLevel;
    }
  };

  const handleError = (error: unknown, context: string): void => {
    console.error(`Error in ${context}:`, error);
    const errorMessage = error instanceof Error ? error.message : `Unknown error in ${context}`;
    setState(prev => ({ ...prev, error: errorMessage, isLoading: false, isAnalyzing: false }));
    toast.error(errorMessage);
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
              
              {!state.analysisComplete && isMounted && !state.showManualInput && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={startAutomaticAnalysis}
                  disabled={state.isLoading}
                  className="inline-flex items-center px-8 py-4 bg-white text-blue-600 font-semibold rounded-full shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {state.isLoading ? (
                    <>
                      <LoadingSpinner className="mr-3" />
                      Detecting Capabilities...
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
                  <div className="flex-1">
                    <h3 className="text-lg font-medium text-red-800 mb-2">Analysis Failed</h3>
                    <p className="text-red-700 mb-4">{state.error}</p>
                    <div className="flex gap-3">
                      <button
                        onClick={startAutomaticAnalysis}
                        className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition-colors"
                      >
                        Try Again
                      </button>
                      <button
                        onClick={restartAnalysis}
                        className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 transition-colors"
                      >
                        Start Over
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Browser Limitation Warning & Manual Input */}
            {state.showManualInput && state.browserDetected && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="mb-8"
              >
                {/* Browser Limitation Explanation */}
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mb-8">
                  <div className="flex">
                    <ExclamationTriangleIcon className="h-6 w-6 text-yellow-400 mr-3 flex-shrink-0" />
                    <div className="flex-1">
                      <h3 className="text-lg font-medium text-yellow-800 mb-2">Browser Security Limitations Detected</h3>
                      <p className="text-yellow-700 mb-4">
                        For security reasons, browsers cannot access detailed hardware information like your actual RAM amount or processor model. 
                        <strong> This is why you see limited/incorrect specs below.</strong>
                      </p>
                      
                      <div className="bg-yellow-100 rounded-lg p-4 mb-4">
                        <h4 className="font-medium text-yellow-800 mb-2">What your browser detected:</h4>
                        <div className="text-sm text-yellow-700 space-y-1">
                          <div>• OS: {state.browserDetected.os}</div>
                          <div>• CPU Threads: {state.browserDetected.cpuCores}</div>
                          <div>• Platform: {state.browserDetected.architecture}</div>
                          {state.browserDetected.gpus && state.browserDetected.gpus.length > 0 && (
                            <div>• GPU: {state.browserDetected.gpus[0].name}</div>
                          )}
                        </div>
                        <p className="text-xs text-yellow-600 mt-2 font-medium">
                          ⚠️ This is likely NOT your actual hardware specs!
                        </p>
                      </div>
                      
                      <p className="text-yellow-700">
                        Please enter your <strong>actual hardware specifications</strong> below for accurate LLM recommendations.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Manual Input Component */}
                <ManualHardwareInput 
                  onComplete={handleManualInput}
                  browserDetected={state.browserDetected}
                />
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
                  We're processing your hardware specifications and checking compatibility 
                  with popular LLM models. This will take just a moment.
                </p>
                {isMounted && typeof navigator !== 'undefined' && (
                  <div className="text-sm text-blue-600 bg-blue-50 rounded-lg p-3 max-w-lg mx-auto">
                    <div className="mb-2"><strong>🔍 Processing your specifications:</strong></div>
                    <div>Platform: {navigator.platform}</div>
                    <div>Browser: {navigator.userAgent.split(' ')[0]}</div>
                    <div>CPU Cores: {navigator.hardwareConcurrency}</div>
                    <div className="text-xs mt-2 text-blue-500">
                      ✅ Analysis running locally in your browser!
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
                      <div className="flex gap-3">
                        <button
                          onClick={() => setState(prev => ({ ...prev, showManualInput: true, analysisComplete: false }))}
                          className="bg-white bg-opacity-20 text-white px-4 py-2 rounded-lg hover:bg-opacity-30 transition-colors text-sm"
                        >
                          Edit Hardware Specs
                        </button>
                        <button
                          onClick={restartAnalysis}
                          className="bg-white bg-opacity-20 text-white px-4 py-2 rounded-lg hover:bg-opacity-30 transition-colors text-sm"
                        >
                          Start Over
                        </button>
                      </div>
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
          {!state.analysisComplete && !state.isAnalyzing && !state.showManualInput && isMounted && (
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
                  We'll detect what we can from your browser, then ask you to confirm your actual hardware 
                  specifications for precise LLM compatibility analysis.
                </p>
              </div>
              
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
                <div className="flex items-center mb-4">
                  <CpuChipIcon className="h-8 w-8 text-purple-500 mr-3" />
                  <h3 className="text-xl font-semibold text-gray-800">Accurate Recommendations</h3>
                </div>
                <p className="text-gray-600">
                  Get personalized model recommendations based on your <strong>actual</strong> system capabilities,
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
                <h3 className="text-lg font-medium text-blue-800 mb-2">Privacy & How It Works</h3>
                <div className="text-blue-700 space-y-2">
                  <p>
                    <strong>🔒 Completely Private:</strong> All analysis happens locally in your browser. 
                    No hardware information is sent to our servers.
                  </p>
                  <p>
                    <strong>🛡️ Browser Limitations:</strong> For security, browsers cannot access your actual RAM, 
                    processor model, or GPU specs. You'll manually enter your real specifications.
                  </p>
                  <p>
                    <strong>✅ Accurate Results:</strong> Manual input ensures you get recommendations 
                    based on your <em>actual</em> hardware, not limited browser detection.
                  </p>
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