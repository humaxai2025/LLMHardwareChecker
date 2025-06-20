#!/usr/bin/env python3
"""
LLM Hardware Compatibility Checker with Installation Guide
Analyzes your system hardware and provides specific installation instructions for compatible LLMs
"""

import platform
import psutil
import subprocess
import sys
from pathlib import Path

# Try to import GPU detection libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class SystemAnalyzer:
    def __init__(self):
        self.system_info = {}
        self.analyze_system()
    
    def analyze_system(self):
        """Analyze system hardware specifications"""
        print("🔍 Analyzing your system hardware...")
        
        # Basic system info
        self.system_info['os'] = platform.system()
        self.system_info['architecture'] = platform.machine()
        self.system_info['processor'] = platform.processor()
        
        # CPU info
        self.system_info['cpu_cores'] = psutil.cpu_count(logical=False)
        self.system_info['cpu_threads'] = psutil.cpu_count(logical=True)
        
        # Memory info
        memory = psutil.virtual_memory()
        self.system_info['total_ram_gb'] = round(memory.total / (1024**3), 1)
        self.system_info['available_ram_gb'] = round(memory.available / (1024**3), 1)
        
        # Storage info
        disk = psutil.disk_usage('/')
        self.system_info['total_storage_gb'] = round(disk.total / (1024**3), 1)
        self.system_info['free_storage_gb'] = round(disk.free / (1024**3), 1)
        
        # GPU info
        self.system_info['gpus'] = self.detect_gpus()
        
    def detect_gpus(self):
        """Detect available GPUs and their VRAM"""
        gpus = []
        
        # Try NVIDIA GPUs first
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpus.append({
                        'name': name,
                        'vram_gb': round(memory_info.total / (1024**3), 1),
                        'type': 'NVIDIA'
                    })
            except Exception as e:
                print(f"⚠️  NVML detection failed: {e}")
        
        # Try GPUtil as fallback
        elif GPU_AVAILABLE:
            try:
                nvidia_gpus = GPUtil.getGPUs()
                for gpu in nvidia_gpus:
                    gpus.append({
                        'name': gpu.name,
                        'vram_gb': round(gpu.memoryTotal / 1024, 1),
                        'type': 'NVIDIA'
                    })
            except Exception as e:
                print(f"⚠️  GPUtil detection failed: {e}")
        
        # Try detecting AMD GPUs (basic detection)
        try:
            if self.system_info['os'] == 'Linux':
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if 'AMD' in result.stdout and 'VGA' in result.stdout:
                    gpus.append({
                        'name': 'AMD GPU (detected)',
                        'vram_gb': 'Unknown',
                        'type': 'AMD'
                    })
        except:
            pass
        
        # Try detecting Apple Silicon
        if self.system_info['architecture'] == 'arm64' and self.system_info['os'] == 'Darwin':
            gpus.append({
                'name': 'Apple Silicon GPU',
                'vram_gb': 'Unified Memory',
                'type': 'Apple'
            })
        
        return gpus
    
    def print_system_info(self):
        """Print detected system information"""
        print("\n" + "="*50)
        print("🖥️  SYSTEM SPECIFICATIONS")
        print("="*50)
        print(f"OS: {self.system_info['os']} ({self.system_info['architecture']})")
        print(f"CPU: {self.system_info['processor']}")
        print(f"CPU Cores: {self.system_info['cpu_cores']} physical / {self.system_info['cpu_threads']} logical")
        print(f"RAM: {self.system_info['total_ram_gb']} GB total ({self.system_info['available_ram_gb']} GB available)")
        print(f"Storage: {self.system_info['free_storage_gb']} GB free / {self.system_info['total_storage_gb']} GB total")
        
        if self.system_info['gpus']:
            print(f"GPUs:")
            for gpu in self.system_info['gpus']:
                vram_info = f"{gpu['vram_gb']} GB VRAM" if isinstance(gpu['vram_gb'], (int, float)) else gpu['vram_gb']
                print(f"  - {gpu['name']} ({vram_info})")
        else:
            print("GPUs: None detected")


class LLMRecommender:
    def __init__(self, system_info):
        self.system_info = system_info
        self.llm_database = self.create_llm_database()
    
    def create_llm_database(self):
        """Database of popular LLMs with installation instructions"""
        return {
            # Small models (1-3B parameters)
            "Phi-3 Mini (3.8B)": {
                "parameters": "3.8B",
                "min_ram_gb": 4,
                "recommended_ram_gb": 8,
                "min_vram_gb": 2,
                "recommended_vram_gb": 4,
                "cpu_only": True,
                "description": "Microsoft's efficient small model, great for basic tasks",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run phi3:mini",
                        "note": "Automatically downloads and runs"
                    },
                    "huggingface": {
                        "model_id": "microsoft/Phi-3-mini-4k-instruct",
                        "command": "from transformers import AutoModelForCausalLM, AutoTokenizer",
                        "note": "Use with transformers library"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf",
                        "recommended_quant": "Q4_K_M",
                        "note": "For llama.cpp and similar tools"
                    }
                }
            },
            "Llama 3.2 3B": {
                "parameters": "3B",
                "min_ram_gb": 4,
                "recommended_ram_gb": 8,
                "min_vram_gb": 2,
                "recommended_vram_gb": 4,
                "cpu_only": True,
                "description": "Meta's latest small model with good performance",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run llama3.2:3b",
                        "note": "Easiest installation method"
                    },
                    "huggingface": {
                        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                        "command": "Requires HF login for access",
                        "note": "Need to accept license on HuggingFace"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Good balance of size and quality"
                    }
                }
            },
            "Gemma 2B": {
                "parameters": "2B",
                "min_ram_gb": 3,
                "recommended_ram_gb": 6,
                "min_vram_gb": 1.5,
                "recommended_vram_gb": 3,
                "cpu_only": True,
                "description": "Google's compact model, very efficient",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run gemma2:2b",
                        "note": "Latest Gemma 2 version"
                    },
                    "huggingface": {
                        "model_id": "google/gemma-2-2b-it",
                        "command": "Requires HF login",
                        "note": "Need to accept license"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Excellent for low-resource systems"
                    }
                }
            },
            
            # Medium models (7-8B parameters)
            "Llama 3.1 8B": {
                "parameters": "8B",
                "min_ram_gb": 8,
                "recommended_ram_gb": 16,
                "min_vram_gb": 5,
                "recommended_vram_gb": 8,
                "cpu_only": True,
                "description": "Excellent balance of performance and efficiency",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run llama3.1:8b",
                        "note": "Most popular choice"
                    },
                    "huggingface": {
                        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        "command": "Requires HF login",
                        "note": "Full precision model"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                        "recommended_quant": "Q4_K_M for 8GB VRAM, Q8_0 for 16GB+",
                        "note": "Choose quantization based on your VRAM"
                    }
                }
            },
            "Mistral 7B": {
                "parameters": "7B",
                "min_ram_gb": 6,
                "recommended_ram_gb": 12,
                "min_vram_gb": 4,
                "recommended_vram_gb": 6,
                "cpu_only": True,
                "description": "High-quality model with strong performance",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run mistral:7b",
                        "note": "Well-optimized by Ollama team"
                    },
                    "huggingface": {
                        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                        "command": "No login required",
                        "note": "Open license model"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "TheBloke's high-quality quantizations"
                    }
                }
            },
            "Code Llama 7B": {
                "parameters": "7B",
                "min_ram_gb": 6,
                "recommended_ram_gb": 12,
                "min_vram_gb": 4,
                "recommended_vram_gb": 6,
                "cpu_only": True,
                "description": "Specialized for code generation and analysis",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run codellama:7b",
                        "note": "For general coding tasks"
                    },
                    "huggingface": {
                        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
                        "command": "Requires HF login",
                        "note": "Instruction-tuned version"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Optimized for coding tasks"
                    }
                }
            },
            
            # Large models (13-15B parameters)
            "Llama 3.1 13B": {
                "parameters": "13B",
                "min_ram_gb": 12,
                "recommended_ram_gb": 24,
                "min_vram_gb": 8,
                "recommended_vram_gb": 12,
                "cpu_only": False,
                "description": "High-quality model requiring more resources",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run llama3.1:13b",
                        "note": "Will download ~7.4GB"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/Meta-Llama-3.1-13B-Instruct-GGUF",
                        "recommended_quant": "Q4_K_M for 8GB VRAM, Q5_K_M for 12GB+",
                        "note": "Requires good GPU or lots of RAM"
                    }
                }
            },
            "Vicuna 13B": {
                "parameters": "13B",
                "min_ram_gb": 12,
                "recommended_ram_gb": 24,
                "min_vram_gb": 8,
                "recommended_vram_gb": 12,
                "cpu_only": False,
                "description": "Fine-tuned for conversations",
                "install_methods": {
                    "huggingface": {
                        "model_id": "lmsys/vicuna-13b-v1.5",
                        "command": "No login required",
                        "note": "Popular chat model"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Good for extended conversations"
                    }
                }
            },
            
            # Extra large models (30B+ parameters)
            "Llama 3.1 70B": {
                "parameters": "70B",
                "min_ram_gb": 48,
                "recommended_ram_gb": 80,
                "min_vram_gb": 40,
                "recommended_vram_gb": 80,
                "cpu_only": False,
                "description": "State-of-the-art performance, requires high-end hardware",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run llama3.1:70b",
                        "note": "Requires 40GB+ free space and powerful hardware"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
                        "recommended_quant": "Q2_K for 24GB VRAM, Q4_K_M for 48GB+",
                        "note": "Consider cloud deployment for this size"
                    }
                }
            },
            "Code Llama 34B": {
                "parameters": "34B",
                "min_ram_gb": 24,
                "recommended_ram_gb": 48,
                "min_vram_gb": 20,
                "recommended_vram_gb": 24,
                "cpu_only": False,
                "description": "Advanced code model for complex programming tasks",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run codellama:34b",
                        "note": "Excellent for complex coding projects"
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF",
                        "recommended_quant": "Q3_K_M for 16GB VRAM, Q4_K_M for 24GB+",
                        "note": "Professional-grade coding assistant"
                    }
                }
            }
        }
    
    def check_compatibility(self, model_name, model_specs):
        """Check if a model is compatible with the current system"""
        compatibility = {
            'can_run_cpu': False,
            'can_run_gpu': False,
            'performance_tier': 'Not Suitable',
            'notes': [],
            'recommended_quant': None
        }
        
        # Check CPU compatibility
        if self.system_info['total_ram_gb'] >= model_specs['min_ram_gb']:
            compatibility['can_run_cpu'] = True
            if self.system_info['total_ram_gb'] >= model_specs['recommended_ram_gb']:
                compatibility['performance_tier'] = 'Good (CPU)'
            else:
                compatibility['performance_tier'] = 'Basic (CPU)'
                compatibility['notes'].append('Consider quantized version for better performance')
        
        # Check GPU compatibility
        if self.system_info['gpus']:
            for gpu in self.system_info['gpus']:
                if isinstance(gpu['vram_gb'], (int, float)):
                    if gpu['vram_gb'] >= model_specs['min_vram_gb']:
                        compatibility['can_run_gpu'] = True
                        if gpu['vram_gb'] >= model_specs['recommended_vram_gb']:
                            compatibility['performance_tier'] = 'Excellent (GPU)'
                            compatibility['recommended_quant'] = 'Q8_0 or FP16'
                        else:
                            compatibility['performance_tier'] = 'Good (GPU)'
                            compatibility['recommended_quant'] = 'Q4_K_M'
                        break
        
        # Add specific notes
        if not compatibility['can_run_cpu'] and not compatibility['can_run_gpu']:
            compatibility['notes'].append('Insufficient RAM/VRAM for this model')
        elif compatibility['can_run_cpu'] and not compatibility['can_run_gpu']:
            compatibility['notes'].append('GPU acceleration not available, will run on CPU')
            compatibility['recommended_quant'] = 'Q4_K_M or Q3_K_M'
        
        return compatibility
    
    def get_recommendations(self):
        """Get LLM recommendations based on system capabilities"""
        recommendations = {
            'excellent': [],
            'good': [],
            'basic': [],
            'not_suitable': []
        }
        
        for model_name, model_specs in self.llm_database.items():
            compatibility = self.check_compatibility(model_name, model_specs)
            
            model_info = {
                'name': model_name,
                'specs': model_specs,
                'compatibility': compatibility
            }
            
            if compatibility['performance_tier'] == 'Excellent (GPU)':
                recommendations['excellent'].append(model_info)
            elif compatibility['performance_tier'] in ['Good (GPU)', 'Good (CPU)']:
                recommendations['good'].append(model_info)
            elif compatibility['performance_tier'] == 'Basic (CPU)':
                recommendations['basic'].append(model_info)
            else:
                recommendations['not_suitable'].append(model_info)
        
        return recommendations
    
    def print_recommendations(self):
        """Print formatted recommendations with installation instructions"""
        recommendations = self.get_recommendations()
        
        print("\n" + "="*50)
        print("🤖 LLM RECOMMENDATIONS & INSTALLATION GUIDE")
        print("="*50)
        
        # Check if no models are suitable
        suitable_models = recommendations['excellent'] + recommendations['good'] + recommendations['basic']
        
        if not suitable_models:
            self.print_insufficient_hardware_message(recommendations)
            return
        
        categories = [
            ('excellent', '🟢 EXCELLENT PERFORMANCE', recommendations['excellent']),
            ('good', '🟡 GOOD PERFORMANCE', recommendations['good']),
            ('basic', '🟠 BASIC PERFORMANCE', recommendations['basic'])
        ]
        
        for category, title, models in categories:
            if models:
                print(f"\n{title}")
                print("-" * len(title))
                for i, model in enumerate(models, 1):
                    self.print_model_details(model, i)
        
        # Show not suitable models briefly
        if recommendations['not_suitable']:
            print(f"\n🔴 NOT RECOMMENDED ({len(recommendations['not_suitable'])} models)")
            print("-" * 30)
            not_suitable_names = [model['name'] for model in recommendations['not_suitable']]
            print(f"Models requiring more resources: {', '.join(not_suitable_names)}")
        
        self.print_installation_platforms()
        self.print_optimization_tips()
    
    def print_insufficient_hardware_message(self, recommendations):
        """Print message when hardware is insufficient for any LLMs"""
        print("\n❌ INSUFFICIENT HARDWARE DETECTED")
        print("="*50)
        
        ram_gb = self.system_info['total_ram_gb']
        storage_gb = self.system_info['free_storage_gb']
        
        print(f"📊 Your current system:")
        print(f"   • RAM: {ram_gb} GB")
        print(f"   • Free Storage: {storage_gb} GB")
        print(f"   • GPUs: {len(self.system_info['gpus'])} detected")
        
        print(f"\n⚠️  Unfortunately, your system doesn't meet the minimum requirements")
        print(f"   for running local LLMs efficiently.")
        
        # Identify specific issues
        issues = []
        if ram_gb < 3:
            issues.append(f"• Insufficient RAM: {ram_gb} GB (minimum 3 GB needed)")
        if storage_gb < 10:
            issues.append(f"• Low storage space: {storage_gb} GB (minimum 10 GB needed)")
        
        if issues:
            print(f"\n🔍 Specific Issues:")
            for issue in issues:
                print(f"   {issue}")
        
        print(f"\n💡 RECOMMENDED SOLUTIONS:")
        print("="*30)
        
        print(f"\n1️⃣  CLOUD-BASED LLM SERVICES (Recommended)")
        print("   🌐 Use online LLM services instead:")
        print("   • ChatGPT (https://chat.openai.com)")
        print("   • Claude (https://claude.ai)")
        print("   • Google Bard (https://bard.google.com)")
        print("   • Perplexity AI (https://perplexity.ai)")
        print("   • Hugging Face Spaces (https://huggingface.co/spaces)")
        print("   ✅ Pros: No hardware requirements, always up-to-date")
        print("   ❌ Cons: Requires internet, may have usage limits")
        
        print(f"\n2️⃣  CLOUD COMPUTING PLATFORMS")
        print("   ☁️  Rent powerful hardware temporarily:")
        print("   • Google Colab (Free tier available)")
        print("   • Kaggle Notebooks (Free GPU hours)")
        print("   • AWS EC2 with GPU instances")
        print("   • Google Cloud Platform")
        print("   • RunPod (GPU rentals)")
        print("   ✅ Pros: Access to powerful hardware")
        print("   ❌ Cons: Costs money (except free tiers)")
        
        print(f"\n3️⃣  HARDWARE UPGRADE OPTIONS")
        print("   🔧 Minimum recommended upgrades:")
        
        if ram_gb < 8:
            print(f"   • RAM: Upgrade to at least 8 GB (16 GB preferred)")
            print(f"     Current: {ram_gb} GB → Target: 8-16 GB")
        
        if storage_gb < 50:
            print(f"   • Storage: Free up space or add storage")
            print(f"     Current: {storage_gb} GB free → Target: 50+ GB free")
        
        if not self.system_info['gpus']:
            print(f"   • GPU: Consider adding a GPU for better performance")
            print(f"     Budget: GTX 1060 6GB / RTX 3060 (8GB VRAM)")
            print(f"     Mid-range: RTX 4060 Ti (16GB VRAM)")
            print(f"     High-end: RTX 4090 (24GB VRAM)")
        
        print(f"\n4️⃣  LIGHTWEIGHT ALTERNATIVES")
        print("   📱 Try these minimal options:")
        print("   • Mobile LLM apps (if you have a smartphone):")
        print("     - LM Studio Mobile")
        print("     - Offline ChatGPT alternatives")
        print("   • Browser-based models:")
        print("     - WebLLM (runs in browser)")
        print("     - Transformers.js demos")
        
        print(f"\n5️⃣  FUTURE PREPARATION")
        print("   📈 When you upgrade your hardware:")
        print("   • Save this script to re-run the analysis")
        print("   • Target specs for good LLM performance:")
        print("     - RAM: 16+ GB")
        print("     - Storage: 100+ GB free")
        print("     - GPU: 8+ GB VRAM (optional but recommended)")
        
        # Show what the smallest model needs
        smallest_model = min(recommendations['not_suitable'], 
                           key=lambda x: x['specs']['min_ram_gb'])
        print(f"\n📏 SMALLEST MODEL REQUIREMENTS:")
        print(f"   Model: {smallest_model['name']}")
        print(f"   Minimum RAM: {smallest_model['specs']['min_ram_gb']} GB")
        print(f"   Your RAM: {ram_gb} GB")
        print(f"   Gap: {smallest_model['specs']['min_ram_gb'] - ram_gb:.1f} GB short")
        
        print(f"\n🎯 IMMEDIATE NEXT STEPS:")
        print("   1. Try cloud-based LLM services (free to start)")
        print("   2. Check if you can free up RAM by closing other programs")
        print("   3. Consider Google Colab for free GPU access")
        print("   4. Plan hardware upgrades if you want local LLMs")
        
        print(f"\n💬 Don't worry! You have many options to work with LLMs.")
        print(f"   Cloud services often provide better performance than")
        print(f"   running smaller models locally anyway!")
    
    def print_model_details(self, model, index):
        """Print detailed model information with installation instructions"""
        print(f"\n{index}. 📦 {model['name']}")
        print(f"   📊 {model['specs']['parameters']} parameters")
        print(f"   📝 {model['specs']['description']}")
        print(f"   💾 RAM: {model['specs']['min_ram_gb']}GB min / {model['specs']['recommended_ram_gb']}GB recommended")
        print(f"   🎮 VRAM: {model['specs']['min_vram_gb']}GB min / {model['specs']['recommended_vram_gb']}GB recommended")
        print(f"   ⚡ Status: {model['compatibility']['performance_tier']}")
        
        if model['compatibility']['recommended_quant']:
            print(f"   🔧 Recommended quantization: {model['compatibility']['recommended_quant']}")
        
        if model['compatibility']['notes']:
            print(f"   ℹ️  Notes: {'; '.join(model['compatibility']['notes'])}")
        
        print(f"   \n   🚀 INSTALLATION OPTIONS:")
        
        # Show installation methods
        install_methods = model['specs']['install_methods']
        
        if 'ollama' in install_methods:
            ollama = install_methods['ollama']
            print(f"   \n   📱 OLLAMA (Recommended for beginners):")
            print(f"      Command: {ollama['command']}")
            print(f"      Note: {ollama['note']}")
        
        if 'huggingface' in install_methods:
            hf = install_methods['huggingface']
            print(f"   \n   🤗 HUGGING FACE:")
            print(f"      Model ID: {hf['model_id']}")
            print(f"      Usage: {hf['command']}")
            print(f"      Note: {hf['note']}")
        
        if 'gguf' in install_methods:
            gguf = install_methods['gguf']
            print(f"   \n   ⚙️  GGUF (llama.cpp compatible):")
            print(f"      Source: {gguf['source']}")
            print(f"      Recommended: {gguf['recommended_quant']}")
            print(f"      Note: {gguf['note']}")
    
    def print_installation_platforms(self):
        """Print detailed platform installation guides"""
        print("\n" + "="*50)
        print("🛠️  INSTALLATION PLATFORMS")
        print("="*50)
        
        os_type = self.system_info['os']
        
        print(f"\n1️⃣  OLLAMA - Easiest Option (Recommended)")
        print("   📥 Installation:")
        if os_type == "Windows":
            print("      • Download installer from https://ollama.ai/download/windows")
            print("      • Run the .exe installer")
            print("      • Open Command Prompt or PowerShell")
        elif os_type == "Darwin":  # macOS
            print("      • Download from https://ollama.ai/download/mac")
            print("      • Or use: brew install ollama")
            print("      • Open Terminal")
        else:  # Linux
            print("      • Run: curl -fsSL https://ollama.ai/install.sh | sh")
            print("      • Or download from https://ollama.ai/download/linux")
            print("      • Open terminal")
        
        print("   🚀 Usage:")
        print("      • ollama run <model-name>")
        print("      • Example: ollama run llama3.1:8b")
        print("      • Models auto-download on first use")
        
        print(f"\n2️⃣  LM STUDIO - GUI Option")
        print("   📥 Installation:")
        print("      • Download from https://lmstudio.ai/")
        print("      • Available for Windows, macOS, and Linux")
        print("      • No command line needed!")
        print("   🎯 Best for: Users who prefer graphical interfaces")
        
        print(f"\n3️⃣  LLAMA.CPP - Advanced Users")
        print("   📥 Installation:")
        if os_type == "Windows":
            print("      • Download release from https://github.com/ggerganov/llama.cpp/releases")
            print("      • Or build from source with Visual Studio")
        elif os_type == "Darwin":
            print("      • brew install llama.cpp")
            print("      • Or build from source: git clone + make")
        else:
            print("      • Build from source: git clone https://github.com/ggerganov/llama.cpp")
            print("      • make -j$(nproc)")
        
        print("   🚀 Usage:")
        print("      • Download GGUF models manually")
        print("      • ./main -m model.gguf -p 'Your prompt here'")
        print("   🎯 Best for: CPU inference and custom setups")
        
        print(f"\n4️⃣  HUGGING FACE TRANSFORMERS - Developers")
        print("   📥 Installation:")
        print("      • pip install transformers torch")
        print("      • For GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   🚀 Usage:")
        print("      • from transformers import AutoModelForCausalLM, AutoTokenizer")
        print("      • model = AutoModelForCausalLM.from_pretrained('model-id')")
        print("   🎯 Best for: Python developers and custom applications")
    
    def print_optimization_tips(self):
        """Print optimization and setup tips based on detected hardware"""
        print("\n" + "="*50)
        print("💡 OPTIMIZATION TIPS FOR YOUR SYSTEM")
        print("="*50)
        
        # Check if any models are suitable first
        recommendations = self.get_recommendations()
        suitable_models = recommendations['excellent'] + recommendations['good'] + recommendations['basic']
        
        if not suitable_models:
            print("⚠️  Since no local LLMs are recommended for your hardware,")
            print("   focus on cloud-based solutions and hardware planning.")
            print("\n🌐 CLOUD OPTIMIZATION TIPS:")
            print("   • Use browser bookmarks for quick access to LLM services")
            print("   • Try free tiers first: ChatGPT, Claude, Bard")
            print("   • Explore Google Colab for learning about LLMs")
            print("   • Consider upgrading RAM as the most cost-effective improvement")
            return
        
        tips = []
        
        # RAM-specific tips
        ram_gb = self.system_info['total_ram_gb']
        if ram_gb < 8:
            tips.append("🔧 LOW RAM OPTIMIZATION:")
            tips.append("   • Use Q3_K_M or Q4_K_M quantized models")
            tips.append("   • Close other applications before running LLMs")
            tips.append("   • Consider using swap file for larger models")
        elif ram_gb < 16:
            tips.append("🔧 MEDIUM RAM OPTIMIZATION:")
            tips.append("   • Q4_K_M quantization works well")
            tips.append("   • Can run 7B models comfortably")
        else:
            tips.append("🔧 HIGH RAM OPTIMIZATION:")
            tips.append("   • Can use Q5_K_M or Q8_0 for better quality")
            tips.append("   • Multiple models can be loaded simultaneously")
        
        # GPU-specific tips
        if self.system_info['gpus']:
            max_vram = max([gpu.get('vram_gb', 0) for gpu in self.system_info['gpus'] if isinstance(gpu.get('vram_gb'), (int, float))], default=0)
            if max_vram > 0:
                if max_vram < 6:
                    tips.append("🎮 GPU OPTIMIZATION (Limited VRAM):")
                    tips.append("   • Use Q4_K_M quantization for GPU inference")
                    tips.append("   • Consider CPU inference for larger models")
                elif max_vram < 12:
                    tips.append("🎮 GPU OPTIMIZATION (Good VRAM):")
                    tips.append("   • Q4_K_M or Q5_K_M work well")
                    tips.append("   • 7B models will run smoothly")
                else:
                    tips.append("🎮 GPU OPTIMIZATION (High VRAM):")
                    tips.append("   • Can use Q8_0 or even FP16 for best quality")
                    tips.append("   • 13B+ models are possible")
            
            # Check for Apple Silicon
            apple_gpu = any(gpu.get('type') == 'Apple' for gpu in self.system_info['gpus'])
            if apple_gpu:
                tips.append("🍎 APPLE SILICON OPTIMIZATION:")
                tips.append("   • Metal acceleration works automatically")
                tips.append("   • Unified memory is shared between CPU and GPU")
                tips.append("   • Ollama has excellent Apple Silicon support")
        else:
            tips.append("💻 CPU-ONLY OPTIMIZATION:")
            tips.append("   • Use llama.cpp for best CPU performance")
            tips.append("   • Enable all CPU cores with -t parameter")
            tips.append("   • Q4_K_M quantization balances speed and quality")
        
        # Storage tips
        free_space = self.system_info['free_storage_gb']
        if free_space < 50:
            tips.append("💾 STORAGE OPTIMIZATION:")
            tips.append("   • Low disk space detected!")
            tips.append("   • Start with smallest models (2B-3B)")
            tips.append("   • Delete unused models regularly")
        
        # General tips
        tips.extend([
            "🚀 GENERAL TIPS:",
            "   • Start with smaller models and work your way up",
            "   • Monitor system resources (htop/Task Manager)",
            "   • Use 'ollama list' to see downloaded models",
            "   • Try different quantization levels to find the sweet spot"
        ])
        
        for tip in tips:
            print(tip)
        
        # Platform-specific notes
        os_type = self.system_info['os']
        if os_type == "Windows":
            print("\n🪟 WINDOWS NOTES:")
            print("   • Windows Defender may scan large model files")
            print("   • Consider using Windows Terminal for better experience")
        elif os_type == "Darwin":
            print("\n🍎 MACOS NOTES:")
            print("   • Metal acceleration provides excellent performance")
            print("   • Models stored in ~/.ollama/models/")
        else:
            print("\n🐧 LINUX NOTES:")
            print("   • Best platform for LLM development")
            print("   • Easy to build tools from source")


def install_requirements():
    """Install required packages if missing"""
    required_packages = ['psutil']
    optional_packages = ['GPUtil', 'nvidia-ml-py3']
    
    missing_required = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    if missing_required:
        print(f"❌ Missing required packages: {missing_required}")
        print("Please install with: pip install " + " ".join(missing_required))
        return False
    
    missing_optional = []
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)
    
    if missing_optional:
        print(f"⚠️  Optional packages not found: {missing_optional}")
        print("For better GPU detection, install with: pip install " + " ".join(missing_optional))
    
    return True


def main():
    """Main function"""
    print("🚀 LLM Hardware Compatibility Checker & Installation Guide")
    print("="*60)
    
    # Check requirements
    if not install_requirements():
        return
    
    # Analyze system
    analyzer = SystemAnalyzer()
    analyzer.print_system_info()
    
    # Get recommendations with installation instructions
    recommender = LLMRecommender(analyzer.system_info)
    recommendations = recommender.get_recommendations()
    recommender.print_recommendations()
    
    # Check if any models are suitable for final message
    suitable_models = recommendations['excellent'] + recommendations['good'] + recommendations['basic']
    
    print("\n" + "="*60)
    if suitable_models:
        print("✅ Analysis complete! Follow the installation instructions above.")
        print("💡 Tip: Start with Ollama if you're new to running local LLMs!")
    else:
        print("📊 Analysis complete! Check the cloud-based alternatives above.")
        print("💡 Tip: Cloud LLM services often perform better than small local models!")
    print("="*60)


if __name__ == "__main__":
    main()