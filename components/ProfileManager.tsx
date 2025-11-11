'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ServerIcon,
  TrashIcon,
  PencilIcon,
  ArrowDownTrayIcon,
  ArrowUpTrayIcon,
  CheckIcon,
  XMarkIcon,
  PlusCircleIcon,
} from '@heroicons/react/24/outline';
import { ProfileManager, HardwareProfile } from '../lib/profileManager';
import { SystemInfo } from '../lib/systemAnalyzer';
import toast from 'react-hot-toast';

interface ProfileManagerComponentProps {
  onSelectProfile: (systemInfo: SystemInfo) => void;
  currentSystemInfo?: SystemInfo | null;
}

const ProfileManagerComponent: React.FC<ProfileManagerComponentProps> = ({
  onSelectProfile,
  currentSystemInfo,
}) => {
  const [profiles, setProfiles] = useState<HardwareProfile[]>([]);
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [saveName, setSaveName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);

  useEffect(() => {
    loadProfiles();
  }, []);

  const loadProfiles = () => {
    const loadedProfiles = ProfileManager.getProfiles();
    setProfiles(loadedProfiles);
    setActiveProfileId(ProfileManager.getActiveProfileId());
  };

  const handleSaveProfile = () => {
    if (!currentSystemInfo) {
      toast.error('No hardware data to save');
      return;
    }

    if (!saveName.trim()) {
      toast.error('Please enter a profile name');
      return;
    }

    try {
      const profile = ProfileManager.saveProfile({
        name: saveName.trim(),
        systemInfo: currentSystemInfo,
      });

      ProfileManager.setActiveProfile(profile.id);
      loadProfiles();
      setSaveName('');
      setShowSaveDialog(false);
      toast.success(`Profile "${profile.name}" saved!`);
    } catch (error) {
      toast.error('Failed to save profile');
    }
  };

  const handleLoadProfile = (profile: HardwareProfile) => {
    ProfileManager.setActiveProfile(profile.id);
    setActiveProfileId(profile.id);
    onSelectProfile(profile.systemInfo);
    toast.success(`Loaded profile: ${profile.name}`);
  };

  const handleDeleteProfile = (id: string, name: string) => {
    if (confirm(`Delete profile "${name}"?`)) {
      ProfileManager.deleteProfile(id);
      loadProfiles();
      toast.success('Profile deleted');
    }
  };

  const handleRenameProfile = (id: string) => {
    if (!editName.trim()) {
      toast.error('Name cannot be empty');
      return;
    }

    ProfileManager.updateProfile(id, { name: editName.trim() });
    loadProfiles();
    setEditingId(null);
    setEditName('');
    toast.success('Profile renamed');
  };

  const handleExportProfile = (id: string, name: string) => {
    try {
      const json = ProfileManager.exportProfile(id);
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${name.replace(/[^a-z0-9]/gi, '_')}_profile.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success('Profile exported');
    } catch (error) {
      toast.error('Failed to export profile');
    }
  };

  const handleImportProfile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = e.target?.result as string;
        const profile = ProfileManager.importProfile(json);
        loadProfiles();
        toast.success(`Imported profile: ${profile.name}`);
      } catch (error) {
        toast.error('Failed to import profile');
      }
    };
    reader.readAsText(file);

    // Reset input
    event.target.value = '';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <ServerIcon className="h-8 w-8 text-blue-500 mr-3" />
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Hardware Profiles</h2>
            <p className="text-sm text-gray-600">Save and manage multiple hardware configurations</p>
          </div>
        </div>

        <div className="flex gap-2">
          {currentSystemInfo && (
            <button
              onClick={() => setShowSaveDialog(true)}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <PlusCircleIcon className="h-5 w-5 mr-2" />
              Save Current
            </button>
          )}

          <label className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors cursor-pointer">
            <ArrowUpTrayIcon className="h-5 w-5 mr-2" />
            Import
            <input
              type="file"
              accept=".json"
              onChange={handleImportProfile}
              className="hidden"
            />
          </label>
        </div>
      </div>

      {/* Save Dialog */}
      <AnimatePresence>
        {showSaveDialog && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg"
          >
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Profile Name
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSaveProfile()}
                placeholder="e.g., Gaming PC, Work Laptop"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                autoFocus
              />
              <button
                onClick={handleSaveProfile}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                <CheckIcon className="h-5 w-5" />
              </button>
              <button
                onClick={() => {
                  setShowSaveDialog(false);
                  setSaveName('');
                }}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Profiles List */}
      {profiles.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <ServerIcon className="h-16 w-16 mx-auto mb-4 opacity-30" />
          <p className="text-lg font-medium">No saved profiles</p>
          <p className="text-sm">Save your hardware configuration to quickly switch between systems</p>
        </div>
      ) : (
        <div className="space-y-3">
          {profiles.map((profile) => (
            <motion.div
              key={profile.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`p-4 border-2 rounded-lg transition-all cursor-pointer ${
                activeProfileId === profile.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300 bg-white'
              }`}
              onClick={() => handleLoadProfile(profile)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  {editingId === profile.id ? (
                    <div className="flex gap-2 mb-2" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleRenameProfile(profile.id)}
                        className="flex-1 px-2 py-1 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                        autoFocus
                      />
                      <button
                        onClick={() => handleRenameProfile(profile.id)}
                        className="p-1 text-green-600 hover:text-green-700"
                      >
                        <CheckIcon className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => {
                          setEditingId(null);
                          setEditName('');
                        }}
                        className="p-1 text-gray-600 hover:text-gray-700"
                      >
                        <XMarkIcon className="h-5 w-5" />
                      </button>
                    </div>
                  ) : (
                    <h3 className="text-lg font-semibold text-gray-800 mb-1">
                      {profile.name}
                      {activeProfileId === profile.id && (
                        <span className="ml-2 text-xs bg-blue-500 text-white px-2 py-1 rounded">Active</span>
                      )}
                    </h3>
                  )}

                  <div className="text-sm text-gray-600 space-y-1">
                    <div>OS: {profile.systemInfo.os} • CPU: {profile.systemInfo.cpuCores} cores</div>
                    <div>RAM: {profile.systemInfo.totalRamGB}GB • GPU: {profile.systemInfo.gpus?.length || 0} detected</div>
                    <div className="text-xs text-gray-400">
                      Created: {new Date(profile.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                </div>

                <div className="flex gap-2 ml-4" onClick={(e) => e.stopPropagation()}>
                  <button
                    onClick={() => {
                      setEditingId(profile.id);
                      setEditName(profile.name);
                    }}
                    className="p-2 text-blue-600 hover:bg-blue-100 rounded-lg transition-colors"
                    title="Rename"
                  >
                    <PencilIcon className="h-5 w-5" />
                  </button>
                  <button
                    onClick={() => handleExportProfile(profile.id, profile.name)}
                    className="p-2 text-green-600 hover:bg-green-100 rounded-lg transition-colors"
                    title="Export"
                  >
                    <ArrowDownTrayIcon className="h-5 w-5" />
                  </button>
                  <button
                    onClick={() => handleDeleteProfile(profile.id, profile.name)}
                    className="p-2 text-red-600 hover:bg-red-100 rounded-lg transition-colors"
                    title="Delete"
                  >
                    <TrashIcon className="h-5 w-5" />
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ProfileManagerComponent;
