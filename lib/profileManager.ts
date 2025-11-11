import { SystemInfo } from './systemAnalyzer';

export interface HardwareProfile {
  id: string;
  name: string;
  systemInfo: SystemInfo;
  createdAt: string;
  updatedAt: string;
}

const PROFILES_STORAGE_KEY = 'llm_hardware_profiles';
const ACTIVE_PROFILE_KEY = 'llm_active_profile';

export class ProfileManager {
  static getProfiles(): HardwareProfile[] {
    if (typeof window === 'undefined') return [];

    try {
      const stored = localStorage.getItem(PROFILES_STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Error loading profiles:', error);
      return [];
    }
  }

  static saveProfile(profile: Omit<HardwareProfile, 'id' | 'createdAt' | 'updatedAt'>): HardwareProfile {
    const profiles = this.getProfiles();
    const now = new Date().toISOString();

    const newProfile: HardwareProfile = {
      ...profile,
      id: `profile_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: now,
      updatedAt: now,
    };

    profiles.push(newProfile);
    localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(profiles));

    return newProfile;
  }

  static updateProfile(id: string, updates: Partial<HardwareProfile>): HardwareProfile | null {
    const profiles = this.getProfiles();
    const index = profiles.findIndex(p => p.id === id);

    if (index === -1) return null;

    profiles[index] = {
      ...profiles[index],
      ...updates,
      updatedAt: new Date().toISOString(),
    };

    localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(profiles));
    return profiles[index];
  }

  static deleteProfile(id: string): boolean {
    const profiles = this.getProfiles();
    const filtered = profiles.filter(p => p.id !== id);

    if (filtered.length === profiles.length) return false;

    localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(filtered));

    // Clear active profile if deleted
    if (this.getActiveProfileId() === id) {
      this.clearActiveProfile();
    }

    return true;
  }

  static getProfile(id: string): HardwareProfile | null {
    const profiles = this.getProfiles();
    return profiles.find(p => p.id === id) || null;
  }

  static setActiveProfile(id: string): void {
    localStorage.setItem(ACTIVE_PROFILE_KEY, id);
  }

  static getActiveProfileId(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(ACTIVE_PROFILE_KEY);
  }

  static getActiveProfile(): HardwareProfile | null {
    const id = this.getActiveProfileId();
    if (!id) return null;
    return this.getProfile(id);
  }

  static clearActiveProfile(): void {
    localStorage.removeItem(ACTIVE_PROFILE_KEY);
  }

  static exportProfile(id: string): string {
    const profile = this.getProfile(id);
    if (!profile) throw new Error('Profile not found');
    return JSON.stringify(profile, null, 2);
  }

  static importProfile(jsonString: string): HardwareProfile {
    try {
      const profile = JSON.parse(jsonString);

      // Validate profile structure
      if (!profile.name || !profile.systemInfo) {
        throw new Error('Invalid profile format');
      }

      // Create new profile with imported data
      return this.saveProfile({
        name: `${profile.name} (Imported)`,
        systemInfo: profile.systemInfo,
      });
    } catch (error) {
      throw new Error('Failed to import profile: ' + (error instanceof Error ? error.message : 'Unknown error'));
    }
  }

  static exportAllProfiles(): string {
    const profiles = this.getProfiles();
    return JSON.stringify(profiles, null, 2);
  }
}
