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
import ManualHardwareInput from '../components/ManualHardwareInput';
import ProfileManagerComponent from '../components/ProfileManager';
import AdvancedToolsTabs from '../components/AdvancedToolsTabs';
import ThemeToggle from '../components/ThemeToggle';
import KeyboardShortcuts from '../components/KeyboardShortcuts';

interface AnalysisState {
  systemInfo: SystemInfo | null;
  recommendations: Recommendations | null;
  recommender: LLMRecommender | null;
  isLoading: boolean;
  isAnalyzing: boolean;
  error: string | null;
  analysisComplete: boolean;
  showManualInput: boolean;
  browserDetected: Partial<SystemInfo> | null;
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
    showManualInput: false,
    browserDetected: null,
  });

  // Ensure we only run client-side code after mount
  useEffect(() => {
    setIsMounted(true);
  }, []);

  const detectBrowserCapabilities = async () => {
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

      // Try to detect GPU
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') as WebGLRenderingContext | null;
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) as string;
          browserDetected.gpus = [{
            name: renderer,
            vramGB: 'Unknown',
            type: 'Detected via WebGL'
          }];
        }
      }

      return browserDetected;
    } catch (error) {
      console.error('Browser detection failed:', error);
      return null;
    }
  };

  const startAutomaticAnalysis = async () => {
    setState(prev => ({ ...prev, isLoading: true, isAnalyzing: true, error: null }));
    
    try {
      toast.loading('Detecting browser capabilities...', { id: 'analysis' });
      
      const browserDetected = await detectBrowserCapabilities();
      
      // Show limitation warning
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
      setState(prev => ({
        ...prev,
        isLoading: false,
        isAnalyzing: false,
        error: 'Failed to detect browser capabilities.',
      }));
      
      toast.error('Detection failed. Please try manual input.', { id: 'analysis' });
    }
  };

  const handleManualInput = async (systemInfo: SystemInfo) => {
    setState(prev => ({ ...prev, isLoading: true, isAnalyzing: true, error: null }));
    
    try {
      toast.loading('Generating recommendations for your system...', { id: 'analysis' });
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      
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
        showManualInput: false,
        browserDetected: null,
      });
      
      toast.success('Analysis complete with your hardware specs!', { id: 'analysis' });
      
    } catch (error) {
      console.error('Analysis failed:', error);
      setState(prev => ({
        ...prev,
        isLoading: false,
        isAnalyzing: false,
        error: `Failed to analyze system: ${error instanceof Error ? error.message : 'Unknown error'}`,
      }));
      
      toast.error('Analysis failed. Please try again.', { id: 'analysis' });
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
      showManualInput: false,
      browserDetected: null,
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
      <div className="min-h-screen bg-gray-50">
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
        
        {/* Professional Header */}
        <header className="relative bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center"
            >
              <div className="flex justify-center mb-6">
                <div className="p-4 bg-blue-50 rounded-xl border border-blue-100">
                  <CpuChipIcon className="h-10 w-10 text-blue-600" />
                </div>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4 tracking-tight">
                LLM Hardware Compatibility Checker
                <span className="ml-3 text-lg font-normal text-blue-600 dark:text-blue-400">v1.0.0</span>
              </h1>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-8 leading-relaxed">
                Professional hardware analysis for running Large Language Models locally.
                Get accurate recommendations tailored to your system specifications.
              </p>
              
              {!state.analysisComplete && !isMounted && (
                <div className="inline-flex items-center px-8 py-4 bg-gray-200 text-gray-600 font-semibold rounded-full">
                  <LoadingSpinner className="mr-3" />
                  Loading...
                </div>
              )}
              
              {!state.analysisComplete && isMounted && !state.showManualInput && (
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={startAutomaticAnalysis}
                  disabled={state.isLoading}
                  className="inline-flex items-center px-8 py-4 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed focus:ring-4 focus:ring-blue-200"
                  aria-label="Start hardware analysis"
                >
                  {state.isLoading ? (
                    <>
                      <LoadingSpinner className="mr-3" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <PlayIcon className="h-5 w-5 mr-3" />
                      Start Analysis
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
                key="error-message"
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
                      onClick={startAutomaticAnalysis}
                      className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition-colors"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Manual Hardware Input */}
            {state.showManualInput && state.browserDetected && (
              <motion.div
                key="manual-input"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="mb-8"
              >
                {/* Manual Input Component */}
                <ManualHardwareInput
                  onComplete={handleManualInput}
                  browserDetected={state.browserDetected}
                />
              </motion.div>
            )}

            {state.isAnalyzing && (
              <motion.div
                key="analyzing"
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
                key="analysis-complete"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ staggerChildren: 0.1 }}
                className="space-y-8"
              >
                {/* Results Summary */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden"
                >
                  <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-5 border-b border-blue-800">
                    <div className="flex items-center justify-between flex-wrap gap-4">
                      <div className="flex items-center">
                        <div className="p-2 bg-white bg-opacity-20 rounded-lg mr-4">
                          <CheckCircleIcon className="h-7 w-7 text-white" />
                        </div>
                        <div>
                          <h2 className="text-2xl font-bold text-white">Analysis Complete</h2>
                          <p className="text-blue-100 text-sm">Comprehensive hardware compatibility results</p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={restartAnalysis}
                          className="px-4 py-2 bg-white bg-opacity-20 hover:bg-opacity-30 text-white text-sm font-medium rounded-lg transition-all focus:ring-2 focus:ring-white focus:ring-opacity-50"
                          aria-label="Edit hardware specifications"
                        >
                          Edit Specs
                        </button>
                        <button
                          onClick={restartAnalysis}
                          className="px-4 py-2 bg-white bg-opacity-20 hover:bg-opacity-30 text-white text-sm font-medium rounded-lg transition-all focus:ring-2 focus:ring-white focus:ring-opacity-50"
                          aria-label="Run analysis again"
                        >
                          Run Again
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

                {/* Advanced Tools Tabs */}
                <AdvancedToolsTabs
                  systemInfo={state.systemInfo}
                  currentCompatibleCount={getSuitableModelsCount()}
                  onSelectProfile={(systemInfo) => handleManualInput(systemInfo)}
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

          {/* Profile Manager - Before Analysis */}
          {!state.analysisComplete && !state.isAnalyzing && !state.showManualInput && isMounted && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-8"
            >
              <ProfileManagerComponent
                onSelectProfile={(systemInfo) => handleManualInput(systemInfo)}
                currentSystemInfo={null}
              />
            </motion.div>
          )}

          {/* Info Section */}
          {!state.analysisComplete && !state.isAnalyzing && !state.showManualInput && isMounted && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8"
            >
              <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200 hover:shadow-md transition-shadow">
                <div className="flex items-start mb-4">
                  <div className="p-3 bg-blue-50 rounded-lg mr-4">
                    <ComputerDesktopIcon className="h-6 w-6 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">System Analysis</h3>
                    <p className="text-sm text-gray-600 leading-relaxed">
                      Enter your hardware specifications and receive instant compatibility analysis
                      for running Large Language Models locally.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200 hover:shadow-md transition-shadow">
                <div className="flex items-start mb-4">
                  <div className="p-3 bg-green-50 rounded-lg mr-4">
                    <CpuChipIcon className="h-6 w-6 text-green-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">Smart Recommendations</h3>
                    <p className="text-sm text-gray-600 leading-relaxed">
                      Get personalized model recommendations based on your system capabilities,
                      from lightweight 2B models to powerful 70B+ models.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200 hover:shadow-md transition-shadow">
                <div className="flex items-start mb-4">
                  <div className="p-3 bg-purple-50 rounded-lg mr-4">
                    <DocumentArrowDownIcon className="h-6 w-6 text-purple-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">Detailed Reports</h3>
                    <p className="text-sm text-gray-600 leading-relaxed">
                      Download comprehensive reports with installation instructions,
                      optimization tips, and platform-specific setup guides.
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Privacy Notice */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-16 bg-white border border-gray-200 rounded-lg p-6 shadow-sm"
          >
            <div className="flex items-start">
              <div className="p-2 bg-blue-50 rounded-lg mr-4 flex-shrink-0">
                <InformationCircleIcon className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Privacy & Security</h3>
                <div className="text-sm text-gray-600 space-y-2">
                  <p>
                    <strong className="text-gray-900">🔒 Client-Side Processing:</strong> All analysis is performed locally in your browser.
                    No hardware information is transmitted to external servers.
                  </p>
                  <p>
                    <strong className="text-gray-900">✅ Accurate Recommendations:</strong> Receive personalized model suggestions
                    based on your actual system specifications and capabilities.
                  </p>
                  <p>
                    <strong className="text-gray-900">📊 Comprehensive Analysis:</strong> Access detailed compatibility reports,
                    installation guides, and optimization recommendations tailored to your hardware.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </main>

        {/* Footer */}
        <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Built with ❤️ by <span className="font-semibold text-blue-600 dark:text-blue-400">Sriram Srinivasan</span>
              </p>
              <p className="text-gray-500 dark:text-gray-500 text-xs mt-2">
                LLM Hardware Compatibility Checker v1.0.0
              </p>
            </div>
          </div>
        </footer>

        {/* Floating Feedback Button */}
        {isMounted && <FeedbackButton />}

        {/* Theme Toggle */}
        {isMounted && <ThemeToggle />}

        {/* Keyboard Shortcuts */}
        {isMounted && <KeyboardShortcuts />}
      </div>
    </ErrorBoundary>
  );
}