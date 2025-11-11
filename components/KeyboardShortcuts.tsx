'use client';

import { useEffect } from 'react';
import { useTheme } from '../lib/ThemeContext';

interface KeyboardShortcutsProps {
  onToggleTheme?: () => void;
}

const KeyboardShortcuts: React.FC<KeyboardShortcutsProps> = () => {
  const { toggleTheme } = useTheme();

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K: Toggle theme
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        toggleTheme();
      }

      // Ctrl/Cmd + P: Print
      if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
        // Let browser handle this naturally
        return;
      }

      // Esc: Close modals/dialogs
      if (e.key === 'Escape') {
        // Can be used by child components
        const event = new CustomEvent('keyboard-escape');
        window.dispatchEvent(event);
      }

      // ? key: Show shortcuts help
      if (e.key === '?' && !e.shiftKey) {
        const event = new CustomEvent('show-shortcuts-help');
        window.dispatchEvent(event);
      }
    };

    window.addEventListener('keydown', handleKeyPress);

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [toggleTheme]);

  return null; // This component doesn't render anything
};

export default KeyboardShortcuts;
