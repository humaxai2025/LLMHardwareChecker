import React, { useState, Fragment } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { XMarkIcon, StarIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline';
import { StarIcon as StarIconSolid } from '@heroicons/react/24/solid';
import { toast } from 'react-hot-toast';

interface FeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface FeedbackData {
  rating: number;
  category: string;
  message: string;
  email: string;
  includeSystemInfo: boolean;
}

const FeedbackModal: React.FC<FeedbackModalProps> = ({ isOpen, onClose }) => {
  const [feedback, setFeedback] = useState<FeedbackData>({
    rating: 0,
    category: '',
    message: '',
    email: '',
    includeSystemInfo: false
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const categories = [
    { value: 'bug', label: '🐛 Bug Report', description: 'Something isn\'t working correctly' },
    { value: 'feature', label: '💡 Feature Request', description: 'Suggest a new feature or improvement' },
    { value: 'ui', label: '🎨 UI/UX Feedback', description: 'Comments about the design or user experience' },
    { value: 'accuracy', label: '📊 Accuracy Issue', description: 'Incorrect hardware detection or recommendations' },
    { value: 'performance', label: '⚡ Performance Issue', description: 'App is slow or unresponsive' },
    { value: 'general', label: '💬 General Feedback', description: 'Other comments or suggestions' }
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!feedback.rating || !feedback.category || !feedback.message.trim()) {
      toast.error('Please fill in all required fields');
      return;
    }

    setIsSubmitting(true);

    try {
      // In a real application, you would send this to your backend
      // For now, we'll simulate the submission and provide helpful information
      
      const submissionData = {
        ...feedback,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
        systemInfo: feedback.includeSystemInfo ? {
          platform: navigator.platform,
          language: navigator.language,
          cookieEnabled: navigator.cookieEnabled,
          onLine: navigator.onLine
        } : null
      };

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // For demo purposes, log to console and show success message
      console.log('Feedback submitted:', submissionData);
      
      toast.success('Thank you for your feedback! We appreciate your input.');
      
      // Reset form
      setFeedback({
        rating: 0,
        category: '',
        message: '',
        email: '',
        includeSystemInfo: false
      });
      
      onClose();
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      toast.error('Failed to submit feedback. Please try again later.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRatingClick = (rating: number) => {
    setFeedback(prev => ({ ...prev, rating }));
  };

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black bg-opacity-25 backdrop-blur-sm" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-md transform overflow-hidden rounded-2xl bg-white p-6 text-left align-middle shadow-xl transition-all">
                <div className="flex items-center justify-between mb-6">
                  <Dialog.Title as="h3" className="text-lg font-medium leading-6 text-gray-900">
                    Share Your Feedback
                  </Dialog.Title>
                  <button
                    onClick={onClose}
                    className="rounded-md text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <XMarkIcon className="h-6 w-6" />
                  </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Rating */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      How would you rate your experience? *
                    </label>
                    <div className="flex space-x-1">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <button
                          key={star}
                          type="button"
                          onClick={() => handleRatingClick(star)}
                          className="p-1 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          {star <= feedback.rating ? (
                            <StarIconSolid className="h-6 w-6 text-yellow-400" />
                          ) : (
                            <StarIcon className="h-6 w-6 text-gray-300 hover:text-yellow-400" />
                          )}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Category */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      What type of feedback is this? *
                    </label>
                    <div className="space-y-2">
                      {categories.map((category) => (
                        <label key={category.value} className="flex items-start cursor-pointer">
                          <input
                            type="radio"
                            name="category"
                            value={category.value}
                            checked={feedback.category === category.value}
                            onChange={(e) => setFeedback(prev => ({ ...prev, category: e.target.value }))}
                            className="mt-1 mr-3 text-blue-600 focus:ring-blue-500"
                          />
                          <div>
                            <div className="text-sm font-medium text-gray-900">{category.label}</div>
                            <div className="text-xs text-gray-500">{category.description}</div>
                          </div>
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Message */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Your feedback *
                    </label>
                    <textarea
                      value={feedback.message}
                      onChange={(e) => setFeedback(prev => ({ ...prev, message: e.target.value }))}
                      rows={4}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                      placeholder="Please share your thoughts, suggestions, or report any issues..."
                      required
                    />
                  </div>

                  {/* Email */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Email (optional)
                    </label>
                    <input
                      type="email"
                      value={feedback.email}
                      onChange={(e) => setFeedback(prev => ({ ...prev, email: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                      placeholder="your@email.com"
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      We'll only use this to follow up on your feedback if needed
                    </p>
                  </div>

                  {/* Include System Info */}
                  <div>
                    <label className="flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={feedback.includeSystemInfo}
                        onChange={(e) => setFeedback(prev => ({ ...prev, includeSystemInfo: e.target.checked }))}
                        className="mr-2 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">
                        Include basic system information to help us debug issues
                      </span>
                    </label>
                    <p className="mt-1 text-xs text-gray-500">
                      This includes browser type, platform, and language settings (no personal data)
                    </p>
                  </div>

                  {/* Submit Button */}
                  <div className="flex space-x-3">
                    <button
                      type="button"
                      onClick={onClose}
                      className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="flex-1 px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                    >
                      {isSubmitting ? (
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <>
                          <PaperAirplaneIcon className="h-4 w-4 mr-2" />
                          Send Feedback
                        </>
                      )}
                    </button>
                  </div>
                </form>

                {/* Privacy Note */}
                <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
                  <p className="text-xs text-blue-800">
                    🔒 Your feedback helps us improve the app. We respect your privacy and will never share your information with third parties.
                  </p>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};

export default FeedbackModal;