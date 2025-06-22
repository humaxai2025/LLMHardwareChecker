// Client-only system analysis wrapper
import { SystemInfo } from './systemAnalyzer';

export const analyzeClientSystem = async (): Promise<SystemInfo> => {
  // Triple check we're in browser
  if (typeof window === 'undefined' || typeof document === 'undefined' || typeof navigator === 'undefined') {
    throw new Error('This function can only run in a browser environment');
  }

  // Dynamic import to ensure no server-side execution
  const { analyzeSystem } = await import('./systemAnalyzer');
  
  return await analyzeSystem();
};