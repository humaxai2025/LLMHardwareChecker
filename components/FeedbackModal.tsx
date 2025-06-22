// FeedbackModal.tsx - Fixed version with EmailJS fallback
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  XMarkIcon,
  PaperAirplaneIcon,
  StarIcon,
  BugAntIcon,
  LightBulbIcon,
  HeartIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { StarIcon as StarSolid } from '@heroicons/react/24/solid';

// Try to import EmailJS, but handle if it's not installed
let emailjs: any = null;
try {
  emailjs = require('@emailjs/browser');
} catch (error) {
  console.log('EmailJS not installed. Feedback will be shown in console.');
}

interface FeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface FeedbackData {
  type: 'bug' | 'feature' | 'general' | 'praise' | '';
  rating: number;
  title: string;
  message: string;
  email: string;
  includeSystemInfo: boolean;
}

const FeedbackModal: React.FC<FeedbackModalProps> = ({ isOpen, onClose }) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [feedback, setFeedback] = useState<FeedbackData>({
    type: '',
    rating: 0,
    title: '',
    message: '',
    email: '',
    includeSystemInfo: true
  });

  // EmailJS Configuration (replace with your actual values when EmailJS is set up)
  const EMAILJS_CONFIG = {
    SERVICE_ID: 'service_rs2ahgp', // Replace with your EmailJS service ID
    TEMPLATE_ID: 'template_b0cyhhi', // Replace with your EmailJS template ID  
    PUBLIC_KEY: 'bjHQxvuwaLl2jgoBo', // Replace with your EmailJS public key
    TO_EMAIL: 'humanxi2025@gmail.com' // Your email address
  };

  const feedbackTypes = [
    {
      id: 'bug',
      label: 'Bug Report',
      icon: BugAntIcon,
      description: 'Something isn\'t working correctly',
      color: 'red'
    },
    {
      id: 'feature',
      label: 'Feature Request',
      icon: LightBulbIcon,
      description: 'Suggest a new feature or improvement',
      color: 'blue'
    },
    {
      id: 'general',
      label: 'General Feedback',
      icon: ExclamationTriangleIcon,
      description: 'Questions, suggestions, or other feedback',
      color: 'yellow'
    },
    {
      id: 'praise',
      label: 'Compliment',
      icon: HeartIcon,
      description: 'Share what you loved about the tool',
      color: 'green'
    }
  ];

  // Check if EmailJS is properly configured
  const isEmailJSConfigured = () => {
    return emailjs && 
           EMAILJS_CONFIG.SERVICE_ID !== 'service_xxxxxxx' && 
           EMAILJS_CONFIG.TEMPLATE_ID !== 'template_xxxxxxx' && 
           EMAILJS_CONFIG.PUBLIC_KEY !== 'xxxxxxxxxxxxxxx';
  };

  const getSystemInfo = () => {
    return {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      screenResolution: `${screen.width}x${screen.height}`,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      timestamp: new Date().toISOString()
    };
  };

  const sendEmailWithEmailJS = async (templateParams: any) => {
    if (!emailjs) {
      throw new Error('EmailJS not available');
    }

    return await emailjs.send(
      EMAILJS_CONFIG.SERVICE_ID,
      EMAILJS_CONFIG.TEMPLATE_ID,
      templateParams,
      EMAILJS_CONFIG.PUBLIC_KEY
    );
  };

  const sendEmailFallback = async (feedbackData: any) => {
    // Fallback method when EmailJS is not available
    // You can implement your own email sending logic here
    // For now, we'll just log to console and show a helpful message
    
    console.log('='.repeat(50));
    console.log('FEEDBACK RECEIVED FOR: humanxi2025@gmail.com');
    console.log('='.repeat(50));
    console.log('Type:', feedbackData.feedback_type);
    console.log('Rating:', feedbackData.rating);
    console.log('From:', feedbackData.from_name);
    console.log('Email:', feedbackData.user_email);
    console.log('Subject:', feedbackData.subject);
    console.log('Message:', feedbackData.message);
    console.log('System Info:', feedbackData.system_info);
    console.log('Timestamp:', feedbackData.timestamp);
    console.log('='.repeat(50));

    // Simulate email sending delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return { status: 200 };
  };

  const handleSubmit = async () => {
    if (!feedback.type || !feedback.message.trim()) {
      toast.error('Please fill in all required fields');
      return;
    }

    setIsSubmitting(true);
    
    try {
      // Prepare email template parameters
      const systemInfo = feedback.includeSystemInfo ? getSystemInfo() : null;
      
      const templateParams = {
        to_email: EMAILJS_CONFIG.TO_EMAIL,
        from_name: feedback.email || 'Anonymous User',
        feedback_type: feedback.type.charAt(0).toUpperCase() + feedback.type.slice(1),
        rating: feedback.rating > 0 ? `${feedback.rating}/5 stars` : 'Not rated',
        subject: feedback.title || `${feedback.type.charAt(0).toUpperCase() + feedback.type.slice(1)} Feedback`,
        message: feedback.message,
        user_email: feedback.email || 'Not provided',
        system_info: systemInfo ? JSON.stringify(systemInfo, null, 2) : 'Not included',
        timestamp: new Date().toLocaleString()
      };

      let response;
      let emailMethod = 'fallback';
      
      if (isEmailJSConfigured()) {
        try {
          // Try EmailJS first
          response = await sendEmailWithEmailJS(templateParams);
          emailMethod = 'emailjs';
        } catch (emailjsError) {
          console.warn('EmailJS failed, falling back to console logging:', emailjsError);
          // Fall back to console logging
          response = await sendEmailFallback(templateParams);
          emailMethod = 'fallback';
        }
      } else {
        // Use fallback method
        response = await sendEmailFallback(templateParams);
        emailMethod = 'fallback';
      }

      if (response.status === 200) {
        // Success messages based on method used
        if (emailMethod === 'emailjs') {
          toast.success('✅ Feedback sent successfully! Thank you for your input.', {
            duration: 5000
          });
        } else {
          toast.success('✅ Feedback recorded! Check browser console for details.', {
            duration: 6000
          });
          
          // Show additional setup info if EmailJS not configured
          if (!emailjs) {
            setTimeout(() => {
              toast('💡 Install EmailJS to send actual emails: npm install @emailjs/browser', {
                duration: 8000,
                icon: '📧'
              });
            }, 1000);
          } else if (!isEmailJSConfigured()) {
            setTimeout(() => {
              toast('⚙️ Configure EmailJS credentials to enable email sending', {
                duration: 8000,
                icon: '📧'
              });
            }, 1000);
          }
        }
        
        onClose();
        resetForm();
      } else {
        throw new Error(`Server responded with status ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to send feedback:', error);
      
      // More specific error messages
      if (error instanceof Error) {
        if (error.message.includes('EmailJS not available')) {
          toast.error('📧 EmailJS not installed. Feedback logged to console instead.', {
            duration: 6000
          });
        } else if (error.message.includes('not properly configured')) {
          toast.error('⚙️ EmailJS not configured. Please set up your EmailJS credentials.', {
            duration: 6000
          });
        } else if (error.message.includes('network') || error.message.includes('fetch')) {
          toast.error('🌐 Network error. Please check your internet connection and try again.', {
            duration: 6000
          });
        } else {
          toast.error(`❌ Failed to send feedback: ${error.message}`, {
            duration: 6000
          });
        }
      } else {
        toast.error('❌ An unexpected error occurred. Please try again later.', {
          duration: 6000
        });
      }
      
      // Don't close modal on error, let user try again
    } finally {
      setIsSubmitting(false);
    }
  };

  const resetForm = () => {
    setFeedback({
      type: '',
      rating: 0,
      title: '',
      message: '',
      email: '',
      includeSystemInfo: true
    });
    setCurrentStep(1);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const canProceedToStep2 = feedback.type !== '';
  const canSubmit = feedback.type !== '' && feedback.message.trim() !== '';

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-50"
            onClick={handleClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-gray-200">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">Share Your Feedback</h2>
                  <p className="text-gray-600 mt-1">Help us improve the LLM Compatibility Checker</p>
                  {!isEmailJSConfigured() && (
                    <div className="mt-2">
                      {!emailjs ? (
                        <div className="flex items-center text-orange-600 text-sm">
                          <span className="w-2 h-2 bg-orange-500 rounded-full mr-2"></span>
                          EmailJS not installed - feedback will be logged to console
                        </div>
                      ) : (
                        <div className="flex items-center text-blue-600 text-sm">
                          <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                          EmailJS installed but requires configuration for email sending
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <button
                  onClick={handleClose}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <XMarkIcon className="h-6 w-6 text-gray-400" />
                </button>
              </div>

              {/* Progress Indicator */}
              <div className="flex items-center justify-center p-4 bg-gray-50">
                <div className="flex items-center space-x-4">
                  <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 ${
                    currentStep >= 1 ? 'bg-blue-600 border-blue-600 text-white' : 'border-gray-300 text-gray-300'
                  }`}>
                    1
                  </div>
                  <div className={`w-12 h-1 ${currentStep >= 2 ? 'bg-blue-600' : 'bg-gray-300'}`} />
                  <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 ${
                    currentStep >= 2 ? 'bg-blue-600 border-blue-600 text-white' : 'border-gray-300 text-gray-300'
                  }`}>
                    2
                  </div>
                </div>
              </div>

              <div className="p-6">
                {/* Step 1: Feedback Type */}
                {currentStep === 1 && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="space-y-6"
                  >
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">
                        What type of feedback do you have?
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {feedbackTypes.map((type) => (
                          <button
                            key={type.id}
                            onClick={() => setFeedback(prev => ({ ...prev, type: type.id as any }))}
                            className={`p-4 border-2 rounded-lg text-left transition-all hover:shadow-md ${
                              feedback.type === type.id
                                ? 'border-blue-500 bg-blue-50'
                                : 'border-gray-200 hover:border-gray-300'
                            }`}
                          >
                            <div className="flex items-center space-x-3">
                              <type.icon className={`h-6 w-6 ${
                                feedback.type === type.id
                                  ? 'text-blue-600'
                                  : 'text-gray-400'
                              }`} />
                              <div>
                                <div className="font-medium text-gray-900">{type.label}</div>
                                <div className="text-sm text-gray-600">{type.description}</div>
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="flex justify-end">
                      <button
                        onClick={() => setCurrentStep(2)}
                        disabled={!canProceedToStep2}
                        className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        Next Step
                      </button>
                    </div>
                  </motion.div>
                )}

                {/* Step 2: Details */}
                {currentStep === 2 && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="space-y-6"
                  >
                    {/* Rating */}
                    <div>
                      <label className="block text-lg font-semibold text-gray-900 mb-3">
                        How would you rate your overall experience? (Optional)
                      </label>
                      <div className="flex items-center space-x-2">
                        {[1, 2, 3, 4, 5].map((star) => (
                          <button
                            key={star}
                            onClick={() => setFeedback(prev => ({ ...prev, rating: star }))}
                            className="p-1 hover:scale-110 transition-transform"
                          >
                            {star <= feedback.rating ? (
                              <StarSolid className="h-8 w-8 text-yellow-400" />
                            ) : (
                              <StarIcon className="h-8 w-8 text-gray-300 hover:text-yellow-400" />
                            )}
                          </button>
                        ))}
                        {feedback.rating > 0 && (
                          <span className="ml-3 text-gray-600">
                            {feedback.rating}/5 stars
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Title */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Title (Optional)
                      </label>
                      <input
                        type="text"
                        value={feedback.title}
                        onChange={(e) => setFeedback(prev => ({ ...prev, title: e.target.value }))}
                        placeholder="Brief summary of your feedback"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>

                    {/* Message */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Your Feedback <span className="text-red-500">*</span>
                      </label>
                      <textarea
                        value={feedback.message}
                        onChange={(e) => setFeedback(prev => ({ ...prev, message: e.target.value }))}
                        placeholder="Please share your detailed feedback, suggestions, or issues..."
                        rows={5}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                        required
                      />
                    </div>

                    {/* Email */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Your Email (Optional)
                      </label>
                      <input
                        type="email"
                        value={feedback.email}
                        onChange={(e) => setFeedback(prev => ({ ...prev, email: e.target.value }))}
                        placeholder="your.email@example.com"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Provide your email if you'd like a response to your feedback
                      </p>
                    </div>

                    {/* System Info Toggle */}
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="includeSystemInfo"
                        checked={feedback.includeSystemInfo}
                        onChange={(e) => setFeedback(prev => ({ ...prev, includeSystemInfo: e.target.checked }))}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="includeSystemInfo" className="ml-2 text-sm text-gray-700">
                        Include system information to help with debugging (browser, OS, etc.)
                      </label>
                    </div>

                    {/* EmailJS Setup Notice */}
                    {!isEmailJSConfigured() && (
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 className="font-semibold text-blue-800 mb-2">📧 Email Configuration Status</h4>
                        {!emailjs ? (
                          <div>
                            <p className="text-sm text-blue-700 mb-2">
                              EmailJS is not installed. To send actual emails:
                            </p>
                            <code className="bg-blue-100 px-2 py-1 rounded text-sm block mb-2">
                              npm install @emailjs/browser
                            </code>
                            <p className="text-xs text-blue-600">
                              For now, feedback will be logged to the browser console.
                            </p>
                          </div>
                        ) : (
                          <div>
                            <p className="text-sm text-blue-700 mb-2">
                              EmailJS is installed but not configured. To enable email sending:
                            </p>
                            <ol className="text-xs text-blue-600 list-decimal list-inside space-y-1">
                              <li>Create account at https://www.emailjs.com/</li>
                              <li>Set up email service (Gmail recommended)</li>
                              <li>Create email template</li>
                              <li>Update EMAILJS_CONFIG in FeedbackModal.tsx</li>
                            </ol>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex justify-between pt-4">
                      <button
                        onClick={() => setCurrentStep(1)}
                        className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        Back
                      </button>
                      <button
                        onClick={handleSubmit}
                        disabled={!canSubmit || isSubmitting}
                        className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
                      >
                        {isSubmitting ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                            <span>Sending...</span>
                          </>
                        ) : (
                          <>
                            <PaperAirplaneIcon className="h-4 w-4" />
                            <span>{emailjs ? 'Send Feedback' : 'Record Feedback'}</span>
                          </>
                        )}
                      </button>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default FeedbackModal;