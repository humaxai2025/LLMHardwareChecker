#!/usr/bin/env python3
"""
Complete LLM Hardware Compatibility Checker with Report Generation
Analyzes your system hardware and provides personalized recommendations for running LLMs locally
Includes HTML and PDF report generation capabilities
"""

import platform
import psutil
import subprocess
import sys
import argparse
import json
import os
from datetime import datetime
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

# For PDF generation using ReportLab
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class SystemAnalyzer:
    def __init__(self):
        self.system_info = {}
        self.analyze_system()

    def analyze_system(self):
        """Analyze system hardware specifications"""
        print("🔍 Analyzing your system hardware...")

        # Basic system info
        self.system_info["os"] = platform.system()
        self.system_info["architecture"] = platform.machine()
        self.system_info["processor"] = platform.processor()

        # CPU info
        self.system_info["cpu_cores"] = psutil.cpu_count(logical=False)
        self.system_info["cpu_threads"] = psutil.cpu_count(logical=True)

        # Memory info
        memory = psutil.virtual_memory()
        self.system_info["total_ram_gb"] = round(memory.total / (1024**3), 1)
        self.system_info["available_ram_gb"] = round(memory.available / (1024**3), 1)

        # Storage info
        disk = psutil.disk_usage("/")
        self.system_info["total_storage_gb"] = round(disk.total / (1024**3), 1)
        self.system_info["free_storage_gb"] = round(disk.free / (1024**3), 1)

        # GPU info
        self.system_info["gpus"] = self.detect_gpus()

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
                    name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)

                    gpus.append(
                        {
                            "name": name,
                            "vram_gb": round(memory_info.total / (1024**3), 1),
                            "type": "NVIDIA",
                        }
                    )
            except Exception as e:
                print(f"⚠️  NVML detection failed: {e}")

        # Try GPUtil as fallback
        elif GPU_AVAILABLE:
            try:
                nvidia_gpus = GPUtil.getGPUs()
                for gpu in nvidia_gpus:
                    gpus.append(
                        {
                            "name": gpu.name,
                            "vram_gb": round(gpu.memoryTotal / 1024, 1),
                            "type": "NVIDIA",
                        }
                    )
            except Exception as e:
                print(f"⚠️  GPUtil detection failed: {e}")

        # Try detecting AMD GPUs (basic detection)
        try:
            if self.system_info["os"] == "Linux":
                result = subprocess.run(["lspci"], capture_output=True, text=True)
                if "AMD" in result.stdout and "VGA" in result.stdout:
                    gpus.append(
                        {
                            "name": "AMD GPU (detected)",
                            "vram_gb": "Unknown",
                            "type": "AMD",
                        }
                    )
        except:
            pass

        # Try detecting Apple Silicon
        if (
            self.system_info["architecture"] == "arm64"
            and self.system_info["os"] == "Darwin"
        ):
            gpus.append(
                {
                    "name": "Apple Silicon GPU",
                    "vram_gb": "Unified Memory",
                    "type": "Apple",
                }
            )

        return gpus

    def print_system_info(self):
        """Print detected system information"""
        print("\n" + "=" * 50)
        print("🖥️  SYSTEM SPECIFICATIONS")
        print("=" * 50)
        print(f"OS: {self.system_info['os']} ({self.system_info['architecture']})")
        print(f"CPU: {self.system_info['processor']}")
        print(
            f"CPU Cores: {self.system_info['cpu_cores']} physical / {self.system_info['cpu_threads']} logical"
        )
        print(
            f"RAM: {self.system_info['total_ram_gb']} GB total ({self.system_info['available_ram_gb']} GB available)"
        )
        print(
            f"Storage: {self.system_info['free_storage_gb']} GB free / {self.system_info['total_storage_gb']} GB total"
        )

        if self.system_info["gpus"]:
            print(f"GPUs:")
            for gpu in self.system_info["gpus"]:
                vram_info = (
                    f"{gpu['vram_gb']} GB VRAM"
                    if isinstance(gpu["vram_gb"], (int, float))
                    else gpu["vram_gb"]
                )
                print(f"  - {gpu['name']} ({vram_info})")
        else:
            print("GPUs: None detected")


class LLMRecommender:
    def __init__(self, system_info):
        self.system_info = system_info
        self.llm_database = self.create_llm_database()

    def create_llm_database(self):
        """Database of popular LLMs with installation instructions"""
        # Note: This covers major, locally-deployable models.
        # For a complete list of all LLMs, see: https://huggingface.co/models
        # Many models (GPT-4, Claude, etc.) are API-only and not included
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
                        "note": "Automatically downloads and runs",
                    },
                    "huggingface": {
                        "model_id": "microsoft/Phi-3-mini-4k-instruct",
                        "command": "from transformers import AutoModelForCausalLM, AutoTokenizer",
                        "note": "Use with transformers library",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf",
                        "recommended_quant": "Q4_K_M",
                        "note": "For llama.cpp and similar tools",
                    },
                },
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
                        "note": "Easiest installation method",
                    },
                    "huggingface": {
                        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                        "command": "Requires HF login for access",
                        "note": "Need to accept license on HuggingFace",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Good balance of size and quality",
                    },
                },
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
                        "note": "Latest Gemma 2 version",
                    },
                    "huggingface": {
                        "model_id": "google/gemma-2-2b-it",
                        "command": "Requires HF login",
                        "note": "Need to accept license",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Excellent for low-resource systems",
                    },
                },
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
                        "note": "Most popular choice",
                    },
                    "huggingface": {
                        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        "command": "Requires HF login",
                        "note": "Full precision model",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                        "recommended_quant": "Q4_K_M for 8GB VRAM, Q8_0 for 16GB+",
                        "note": "Choose quantization based on your VRAM",
                    },
                },
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
                        "note": "Well-optimized by Ollama team",
                    },
                    "huggingface": {
                        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                        "command": "No login required",
                        "note": "Open license model",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "TheBloke's high-quality quantizations",
                    },
                },
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
                        "note": "For general coding tasks",
                    },
                    "huggingface": {
                        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
                        "command": "Requires HF login",
                        "note": "Instruction-tuned version",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Optimized for coding tasks",
                    },
                },
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
                        "note": "Will download ~7.4GB",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/Meta-Llama-3.1-13B-Instruct-GGUF",
                        "recommended_quant": "Q4_K_M for 8GB VRAM, Q5_K_M for 12GB+",
                        "note": "Requires good GPU or lots of RAM",
                    },
                },
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
                        "note": "Popular chat model",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Good for extended conversations",
                    },
                },
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
                        "note": "Requires 40GB+ free space and powerful hardware",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
                        "recommended_quant": "Q2_K for 24GB VRAM, Q4_K_M for 48GB+",
                        "note": "Consider cloud deployment for this size",
                    },
                },
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
                        "note": "Excellent for complex coding projects",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF",
                        "recommended_quant": "Q3_K_M for 16GB VRAM, Q4_K_M for 24GB+",
                        "note": "Professional-grade coding assistant",
                    },
                },
            },
            # SPECIALIZED MODELS - Domain-specific, truly deployable models
            "StarCoder 7B": {
                "parameters": "7B",
                "min_ram_gb": 6,
                "recommended_ram_gb": 12,
                "min_vram_gb": 4,
                "recommended_vram_gb": 6,
                "cpu_only": True,
                "domain": "Code Generation",
                "description": "Multi-language code generation, supports 80+ programming languages",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run starcoder:7b",
                        "note": "Excellent for diverse programming languages",
                    },
                    "huggingface": {
                        "model_id": "bigcode/starcoder",
                        "command": "No login required",
                        "note": "Open source code generation model",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/starcoder-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Optimized for code completion and generation",
                    },
                },
            },
            "BioMistral 7B": {
                "parameters": "7B",
                "min_ram_gb": 6,
                "recommended_ram_gb": 12,
                "min_vram_gb": 4,
                "recommended_vram_gb": 6,
                "cpu_only": True,
                "domain": "Medical/Biology",
                "description": "Medical knowledge model for healthcare professionals and researchers",
                "install_methods": {
                    "huggingface": {
                        "model_id": "BioMistral/BioMistral-7B",
                        "command": "No login required",
                        "note": "Specialized for medical and biological text",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/BioMistral-7B-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Optimized for medical Q&A and research",
                    },
                },
            },
            "MetaMath 7B": {
                "parameters": "7B",
                "min_ram_gb": 6,
                "recommended_ram_gb": 12,
                "min_vram_gb": 4,
                "recommended_vram_gb": 6,
                "cpu_only": True,
                "domain": "Mathematics",
                "description": "Mathematical reasoning and problem-solving specialist",
                "install_methods": {
                    "huggingface": {
                        "model_id": "meta-math/MetaMath-7B-V1.0",
                        "command": "No login required",
                        "note": "Excellent for mathematical problem solving",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/MetaMath-7B-V1.0-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Optimized for step-by-step math solutions",
                    },
                },
            },
            "WizardCoder 7B": {
                "parameters": "7B",
                "min_ram_gb": 6,
                "recommended_ram_gb": 12,
                "min_vram_gb": 4,
                "recommended_vram_gb": 6,
                "cpu_only": True,
                "domain": "Code Generation",
                "description": "Advanced coding assistant with strong reasoning capabilities",
                "install_methods": {
                    "huggingface": {
                        "model_id": "WizardLM/WizardCoder-7B-V1.0",
                        "command": "No login required",
                        "note": "Enhanced code generation and debugging",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/WizardCoder-7B-V1.0-GGUF",
                        "recommended_quant": "Q4_K_M",
                        "note": "Strong performance on coding benchmarks",
                    },
                },
            },
            "Nous Hermes 2 - Solar 10.7B": {
                "parameters": "10.7B",
                "min_ram_gb": 8,
                "recommended_ram_gb": 16,
                "min_vram_gb": 6,
                "recommended_vram_gb": 8,
                "cpu_only": True,
                "domain": "Research/Analysis",
                "description": "Research and analysis specialist with strong reasoning",
                "install_methods": {
                    "ollama": {
                        "command": "ollama run nous-hermes2-solar:10.7b",
                        "note": "Excellent for research and analytical tasks",
                    },
                    "huggingface": {
                        "model_id": "NousResearch/Nous-Hermes-2-SOLAR-10.7B",
                        "command": "No login required",
                        "note": "High-quality reasoning and analysis",
                    },
                    "gguf": {
                        "source": "https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF",
                        "recommended_quant": "Q4_K_M for 8GB VRAM, Q5_K_M for 12GB+",
                        "note": "Great for complex analytical tasks",
                    },
                },
            },
        }

    def check_compatibility(self, model_name, model_specs):
        """Check if a model is compatible with the current system"""
        compatibility = {
            "can_run_cpu": False,
            "can_run_gpu": False,
            "performance_tier": "Not Suitable",
            "notes": [],
            "recommended_quant": None,
        }

        # Check CPU compatibility
        if self.system_info["total_ram_gb"] >= model_specs["min_ram_gb"]:
            compatibility["can_run_cpu"] = True
            if self.system_info["total_ram_gb"] >= model_specs["recommended_ram_gb"]:
                compatibility["performance_tier"] = "Good (CPU)"
            else:
                compatibility["performance_tier"] = "Basic (CPU)"
                compatibility["notes"].append(
                    "Consider quantized version for better performance"
                )

        # Check GPU compatibility
        if self.system_info["gpus"]:
            for gpu in self.system_info["gpus"]:
                if isinstance(gpu["vram_gb"], (int, float)):
                    if gpu["vram_gb"] >= model_specs["min_vram_gb"]:
                        compatibility["can_run_gpu"] = True
                        if gpu["vram_gb"] >= model_specs["recommended_vram_gb"]:
                            compatibility["performance_tier"] = "Excellent (GPU)"
                            compatibility["recommended_quant"] = "Q8_0 or FP16"
                        else:
                            compatibility["performance_tier"] = "Good (GPU)"
                            compatibility["recommended_quant"] = "Q4_K_M"
                        break

        # Add specific notes
        if not compatibility["can_run_cpu"] and not compatibility["can_run_gpu"]:
            compatibility["notes"].append("Insufficient RAM/VRAM for this model")
        elif compatibility["can_run_cpu"] and not compatibility["can_run_gpu"]:
            compatibility["notes"].append(
                "GPU acceleration not available, will run on CPU"
            )
            compatibility["recommended_quant"] = "Q4_K_M or Q3_K_M"

        return compatibility

    def get_recommendations(self):
        """Get LLM recommendations based on system capabilities"""
        recommendations = {"excellent": [], "good": [], "basic": [], "not_suitable": []}

        for model_name, model_specs in self.llm_database.items():
            compatibility = self.check_compatibility(model_name, model_specs)

            model_info = {
                "name": model_name,
                "specs": model_specs,
                "compatibility": compatibility,
            }

            if compatibility["performance_tier"] == "Excellent (GPU)":
                recommendations["excellent"].append(model_info)
            elif compatibility["performance_tier"] in ["Good (GPU)", "Good (CPU)"]:
                recommendations["good"].append(model_info)
            elif compatibility["performance_tier"] == "Basic (CPU)":
                recommendations["basic"].append(model_info)
            else:
                recommendations["not_suitable"].append(model_info)

        return recommendations

    def print_recommendations(self):
        """Print formatted recommendations with installation instructions"""
        recommendations = self.get_recommendations()

        print("\n" + "=" * 50)
        print("🤖 LLM RECOMMENDATIONS & INSTALLATION GUIDE")
        print("=" * 50)

        # Check if no models are suitable
        suitable_models = (
            recommendations["excellent"]
            + recommendations["good"]
            + recommendations["basic"]
        )

        if not suitable_models:
            self.print_insufficient_hardware_message(recommendations)
            return

        # Separate general and specialized models
        general_models = {"excellent": [], "good": [], "basic": []}
        specialized_models = {"excellent": [], "good": [], "basic": []}

        for category in ["excellent", "good", "basic"]:
            for model in recommendations[category]:
                if "domain" in model["specs"]:
                    specialized_models[category].append(model)
                else:
                    general_models[category].append(model)

        # Print general models first
        print("\n🌟 GENERAL PURPOSE MODELS")
        print("=" * 40)

        general_categories = [
            ("excellent", "🟢 EXCELLENT PERFORMANCE", general_models["excellent"]),
            ("good", "🟡 GOOD PERFORMANCE", general_models["good"]),
            ("basic", "🟠 BASIC PERFORMANCE", general_models["basic"]),
        ]

        general_model_count = 0
        for category, title, models in general_categories:
            if models:
                print(f"\n{title}")
                print("-" * len(title))
                for i, model in enumerate(models, general_model_count + 1):
                    self.print_model_details(model, i)
                general_model_count += len(models)

        # Print specialized models if any are suitable
        specialized_count = sum(
            len(specialized_models[cat]) for cat in ["excellent", "good", "basic"]
        )
        if specialized_count > 0:
            print(f"\n🔬 SPECIALIZED DOMAIN MODELS")
            print("=" * 40)
            print("These models are optimized for specific use cases and domains:")

            specialized_categories = [
                (
                    "excellent",
                    "🟢 EXCELLENT PERFORMANCE",
                    specialized_models["excellent"],
                ),
                ("good", "🟡 GOOD PERFORMANCE", specialized_models["good"]),
                ("basic", "🟠 BASIC PERFORMANCE", specialized_models["basic"]),
            ]

            for category, title, models in specialized_categories:
                if models:
                    print(f"\n{title}")
                    print("-" * len(title))
                    for i, model in enumerate(models, general_model_count + 1):
                        self.print_model_details(model, i)
                    general_model_count += len(models)

        # Show not suitable models briefly
        if recommendations["not_suitable"]:
            print(
                f"\n🔴 NOT RECOMMENDED ({len(recommendations['not_suitable'])} models)"
            )
            print("-" * 30)
            not_suitable_names = [
                model["name"] for model in recommendations["not_suitable"]
            ]
            print(f"Models requiring more resources: {', '.join(not_suitable_names)}")

        self.print_installation_platforms()
        self.print_optimization_tips()
        self.print_additional_models_info()

    def print_insufficient_hardware_message(self, recommendations):
        """Print message when hardware is insufficient for any LLMs"""
        print("\n❌ INSUFFICIENT HARDWARE DETECTED")
        print("=" * 50)

        ram_gb = self.system_info["total_ram_gb"]
        storage_gb = self.system_info["free_storage_gb"]

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
            issues.append(
                f"• Low storage space: {storage_gb} GB (minimum 10 GB needed)"
            )

        if issues:
            print(f"\n🔍 Specific Issues:")
            for issue in issues:
                print(f"   {issue}")

        print(f"\n💡 RECOMMENDED SOLUTIONS:")
        print("=" * 30)

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

        if not self.system_info["gpus"]:
            print(f"   • GPU: Consider adding a GPU for better performance")
            print(f"     Budget: GTX 1060 6GB / RTX 3060 (8GB VRAM)")
            print(f"     Mid-range: RTX 4060 Ti (16GB VRAM)")
            print(f"     High-end: RTX 4090 (24GB VRAM)")

        # Show what the smallest model needs
        smallest_model = min(
            recommendations["not_suitable"], key=lambda x: x["specs"]["min_ram_gb"]
        )
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

        # Show domain for specialized models
        if "domain" in model["specs"]:
            print(f"   🎯 Domain: {model['specs']['domain']}")

        print(f"   📊 {model['specs']['parameters']} parameters")
        print(f"   📝 {model['specs']['description']}")
        print(
            f"   💾 RAM: {model['specs']['min_ram_gb']}GB min / {model['specs']['recommended_ram_gb']}GB recommended"
        )
        print(
            f"   🎮 VRAM: {model['specs']['min_vram_gb']}GB min / {model['specs']['recommended_vram_gb']}GB recommended"
        )
        print(f"   ⚡ Status: {model['compatibility']['performance_tier']}")

        if model["compatibility"]["recommended_quant"]:
            print(
                f"   🔧 Recommended quantization: {model['compatibility']['recommended_quant']}"
            )

        if model["compatibility"]["notes"]:
            print(f"   ℹ️  Notes: {'; '.join(model['compatibility']['notes'])}")

        print(f"   \n   🚀 INSTALLATION OPTIONS:")

        # Show installation methods
        install_methods = model["specs"]["install_methods"]

        if "ollama" in install_methods:
            ollama = install_methods["ollama"]
            print(f"   \n   📱 OLLAMA (Recommended for beginners):")
            print(f"      Command: {ollama['command']}")
            print(f"      Note: {ollama['note']}")

        if "huggingface" in install_methods:
            hf = install_methods["huggingface"]
            print(f"   \n   🤗 HUGGING FACE:")
            print(f"      Model ID: {hf['model_id']}")
            print(f"      Usage: {hf['command']}")
            print(f"      Note: {hf['note']}")

        if "gguf" in install_methods:
            gguf = install_methods["gguf"]
            print(f"   \n   ⚙️  GGUF (llama.cpp compatible):")
            print(f"      Source: {gguf['source']}")
            print(f"      Recommended: {gguf['recommended_quant']}")
            print(f"      Note: {gguf['note']}")

    def print_installation_platforms(self):
        """Print detailed platform installation guides"""
        print("\n" + "=" * 50)
        print("🛠️  INSTALLATION PLATFORMS")
        print("=" * 50)

        os_type = self.system_info["os"]

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
            print(
                "      • Download release from https://github.com/ggerganov/llama.cpp/releases"
            )
            print("      • Or build from source with Visual Studio")
        elif os_type == "Darwin":
            print("      • brew install llama.cpp")
            print("      • Or build from source: git clone + make")
        else:
            print(
                "      • Build from source: git clone https://github.com/ggerganov/llama.cpp"
            )
            print("      • make -j$(nproc)")

        print("   🚀 Usage:")
        print("      • Download GGUF models manually")
        print("      • ./main -m model.gguf -p 'Your prompt here'")
        print("   🎯 Best for: CPU inference and custom setups")

        print(f"\n4️⃣  HUGGING FACE TRANSFORMERS - Developers")
        print("   📥 Installation:")
        print("      • pip install transformers torch")
        print(
            "      • For GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )
        print("   🚀 Usage:")
        print("      • from transformers import AutoModelForCausalLM, AutoTokenizer")
        print("      • model = AutoModelForCausalLM.from_pretrained('model-id')")
        print("   🎯 Best for: Python developers and custom applications")

    def print_optimization_tips(self):
        """Print optimization and setup tips based on detected hardware"""
        print("\n" + "=" * 50)
        print("💡 OPTIMIZATION TIPS FOR YOUR SYSTEM")
        print("=" * 50)

        # Check if any models are suitable first
        recommendations = self.get_recommendations()
        suitable_models = (
            recommendations["excellent"]
            + recommendations["good"]
            + recommendations["basic"]
        )

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
        ram_gb = self.system_info["total_ram_gb"]
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
        if self.system_info["gpus"]:
            max_vram = max(
                [
                    gpu.get("vram_gb", 0)
                    for gpu in self.system_info["gpus"]
                    if isinstance(gpu.get("vram_gb"), (int, float))
                ],
                default=0,
            )
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
            apple_gpu = any(
                gpu.get("type") == "Apple" for gpu in self.system_info["gpus"]
            )
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
        free_space = self.system_info["free_storage_gb"]
        if free_space < 50:
            tips.append("💾 STORAGE OPTIMIZATION:")
            tips.append("   • Low disk space detected!")
            tips.append("   • Start with smallest models (2B-3B)")
            tips.append("   • Delete unused models regularly")

        # General tips
        tips.extend(
            [
                "🚀 GENERAL TIPS:",
                "   • Start with smaller models and work your way up",
                "   • Monitor system resources (htop/Task Manager)",
                "   • Use 'ollama list' to see downloaded models",
                "   • Try different quantization levels to find the sweet spot",
            ]
        )

        for tip in tips:
            print(tip)

        # Platform-specific notes
        os_type = self.system_info["os"]
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

    def print_additional_models_info(self):
        """Information about finding additional models not covered in this script"""
        print("\n" + "=" * 50)
        print("🔍 LOOKING FOR MORE MODELS?")
        print("=" * 50)

        print("This script covers popular general-purpose models and selected")
        print("specialized models for common domains. For additional models:")

        print("\n📚 MODEL REPOSITORIES:")
        print("   • HuggingFace: https://huggingface.co/models")
        print("     - Filter by 'Text Generation' and sort by downloads")
        print("     - Look for models with GGUF versions available")

        print("   • Ollama Library: https://ollama.ai/library")
        print("     - Curated models optimized for local deployment")

        print("   • TheBloke's GGUF: https://huggingface.co/TheBloke")
        print("     - High-quality quantized versions of popular models")

        print("\n🆕 ADDITIONAL SPECIALIZED DOMAINS:")
        print("   • Legal: LegalBERT, LawGPT variants")
        print("   • Finance: FinGPT variants, BloombergGPT")
        print("   • Scientific: SciBERT, Galactica variants")
        print("   • Multilingual: aya-101, XGLM variants")
        print("   • Creative Writing: Specialized creative models")

        print("\n🔬 MORE DOMAIN-SPECIFIC MODELS:")
        print("   We include these specialized domains:")

        # Show which domains are already covered
        domains_covered = set()
        for model_name, model_specs in self.llm_database.items():
            if "domain" in model_specs:
                domains_covered.add(model_specs["domain"])

        if domains_covered:
            print(f"   ✅ Covered: {', '.join(sorted(domains_covered))}")

        print("   🔍 Still exploring: Legal, Finance, Scientific Research")
        print("   💡 Suggest more at: GitHub Issues")

        print("\n💡 FINDING COMPATIBLE MODELS:")
        print("   Use your system specs as a guide:")
        ram_gb = self.system_info["total_ram_gb"]
        max_vram = max(
            [
                gpu.get("vram_gb", 0)
                for gpu in self.system_info["gpus"]
                if isinstance(gpu.get("vram_gb"), (int, float))
            ],
            default=0,
        )

        if ram_gb >= 32 and max_vram >= 16:
            print("   • Your system can handle: Up to 30B models")
        elif ram_gb >= 16 and max_vram >= 8:
            print("   • Your system can handle: Up to 13B models")
        elif ram_gb >= 8:
            print("   • Your system can handle: Up to 7B models")
        else:
            print("   • Your system can handle: Up to 3B models")

        print("   • Look for Q4_K_M or Q5_K_M quantized versions")
        print("   • Check if model has Ollama support for easy installation")

        print("\n⚠️  MODELS NOT SUITABLE FOR LOCAL DEPLOYMENT:")
        print("   • GPT-4, GPT-3.5 (OpenAI API only)")
        print("   • Claude 3/4 (Anthropic API only)")
        print("   • Gemini (Google API only)")
        print("   • Most commercial/proprietary models")

    def generate_report(self, format_type="both", output_dir="."):
        """Generate HTML and/or PDF reports of the analysis"""
        recommendations = self.get_recommendations()
        report_generator = ReportGenerator(self.system_info, recommendations, self)

        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type in ["html", "both"]:
            html_path = os.path.join(
                output_dir, f"llm_compatibility_report_{timestamp}.html"
            )
            html_file = report_generator.generate_html_report(html_path)
            results["html"] = html_file
            print(f"📄 HTML report generated: {html_file}")

        if format_type in ["pdf", "both"]:
            if not REPORTLAB_AVAILABLE:
                print("⚠️  PDF generation requires ReportLab: pip install reportlab")
                results["pdf"] = None
            else:
                pdf_path = os.path.join(
                    output_dir, f"llm_compatibility_report_{timestamp}.pdf"
                )
                pdf_file = report_generator.generate_pdf_report(pdf_path)
                results["pdf"] = pdf_file
                print(f"📄 PDF report generated: {pdf_file}")

        return results


# Fix for SyntaxError: 'return' outside function
#
# The issue is that the methods need to be properly indented inside the ReportGenerator class
# Here's the correct way to integrate them:


class ReportGenerator:
    def __init__(self, system_info, recommendations, recommender):
        self.system_info = system_info
        self.recommendations = recommendations
        self.recommender = recommender
        self.timestamp = datetime.now()

    def generate_html_report(self, output_path="llm_compatibility_report.html"):
        """Generate a detailed HTML report"""
        html_content = self._create_html_content()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

    def generate_pdf_report(self, output_path="llm_compatibility_report.pdf"):
        """Generate a PDF report using ReportLab with installation instructions"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab not available. Install with: pip install reportlab"
            )

        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Build story (content)
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.HexColor("#2c3e50"),
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor("#3498db"),
        )

        subheading_style = ParagraphStyle(
            "CustomSubHeading",
            parent=styles["Heading3"],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor("#2c3e50"),
        )

        code_style = ParagraphStyle(
            "CodeStyle",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=9,
            backgroundColor=colors.HexColor("#f8f9fa"),
            borderPadding=8,
            leftIndent=20,
        )

        # Title
        story.append(Paragraph("🤖 LLM Hardware Compatibility Report", title_style))
        story.append(
            Paragraph(
                f"Generated on {self.timestamp.strftime('%B %d, %Y at %I:%M %p')}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 20))

        # System Specifications
        story.append(Paragraph("🖥️ System Specifications", heading_style))

        # Create system specs table
        sys_data = [
            ["Component", "Specification"],
            [
                "Operating System",
                f"{self.system_info['os']} ({self.system_info['architecture']})",
            ],
            ["Processor", self.system_info["processor"]],
            [
                "CPU Cores",
                f"{self.system_info['cpu_cores']} physical / {self.system_info['cpu_threads']} logical",
            ],
            [
                "Memory (RAM)",
                f"{self.system_info['total_ram_gb']} GB total ({self.system_info['available_ram_gb']} GB available)",
            ],
            [
                "Storage",
                f"{self.system_info['free_storage_gb']} GB free / {self.system_info['total_storage_gb']} GB total",
            ],
        ]

        if self.system_info["gpus"]:
            for i, gpu in enumerate(self.system_info["gpus"]):
                vram_info = (
                    f"{gpu['vram_gb']} GB VRAM"
                    if isinstance(gpu["vram_gb"], (int, float))
                    else gpu["vram_gb"]
                )
                sys_data.append([f"GPU {i+1}", f"{gpu['name']} ({vram_info})"])
        else:
            sys_data.append(["GPU", "None detected"])

        sys_table = Table(sys_data, colWidths=[2 * inch, 4 * inch])
        sys_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        story.append(sys_table)
        story.append(Spacer(1, 20))

        # Recommendations
        suitable_models = (
            self.recommendations["excellent"]
            + self.recommendations["good"]
            + self.recommendations["basic"]
        )

        if not suitable_models:
            story.append(Paragraph("❌ Insufficient Hardware Detected", heading_style))
            story.append(
                Paragraph(
                    "Your system doesn't meet the minimum requirements for running local LLMs efficiently.",
                    styles["Normal"],
                )
            )
            story.append(Spacer(1, 12))
            story.append(
                Paragraph("Recommended cloud-based solutions:", styles["Heading3"])
            )
            story.append(
                Paragraph("• ChatGPT: https://chat.openai.com", styles["Normal"])
            )
            story.append(Paragraph("• Claude: https://claude.ai", styles["Normal"]))
            story.append(
                Paragraph("• Google Bard: https://bard.google.com", styles["Normal"])
            )
        else:
            story.append(
                Paragraph(
                    "🤖 Model Recommendations & Installation Guide", heading_style
                )
            )

            # Summary statistics
            total_specialized = sum(
                len([m for m in self.recommendations[cat] if "domain" in m["specs"]])
                for cat in ["excellent", "good", "basic"]
            )
            total_general = len(suitable_models) - total_specialized

            story.append(
                Paragraph(
                    f"Compatible Models: {len(suitable_models)} | "
                    + f"General Purpose: {total_general} | "
                    + f"Specialized: {total_specialized}",
                    styles["Normal"],
                )
            )
            story.append(Spacer(1, 12))

            # Add model recommendations with installation instructions
            categories = [
                (
                    "excellent",
                    "🟢 Excellent Performance",
                    self.recommendations["excellent"],
                ),
                ("good", "🟡 Good Performance", self.recommendations["good"]),
                ("basic", "🟠 Basic Performance", self.recommendations["basic"]),
            ]

            for category, title, models in categories:
                if models:
                    story.append(Paragraph(title, subheading_style))

                    for model in models:
                        # Model details
                        model_name = model["name"]
                        specs = model["specs"]
                        compatibility = model["compatibility"]

                        domain_text = (
                            f" (Domain: {specs['domain']})" if "domain" in specs else ""
                        )
                        story.append(
                            Paragraph(
                                f"<b>{model_name}</b>{domain_text}", styles["Normal"]
                            )
                        )
                        story.append(
                            Paragraph(
                                f"Parameters: {specs['parameters']} | "
                                + f"RAM: {specs['min_ram_gb']}-{specs['recommended_ram_gb']} GB | "
                                + f"VRAM: {specs['min_vram_gb']}-{specs['recommended_vram_gb']} GB",
                                styles["Normal"],
                            )
                        )
                        story.append(Paragraph(specs["description"], styles["Normal"]))

                        # Installation Instructions
                        story.append(
                            Paragraph("<b>Installation Options:</b>", styles["Normal"])
                        )

                        install_methods = specs["install_methods"]

                        if "ollama" in install_methods:
                            ollama = install_methods["ollama"]
                            story.append(
                                Paragraph(
                                    f"<b>📱 Ollama (Recommended):</b>", styles["Normal"]
                                )
                            )
                            story.append(Paragraph(f"{ollama['command']}", code_style))
                            story.append(
                                Paragraph(
                                    f"Setup: Install from ollama.ai, then run command above. {ollama['note']}",
                                    styles["Normal"],
                                )
                            )

                        if "huggingface" in install_methods:
                            hf = install_methods["huggingface"]
                            story.append(
                                Paragraph(f"<b>🤗 HuggingFace:</b>", styles["Normal"])
                            )
                            story.append(
                                Paragraph(f"Model ID: {hf['model_id']}", code_style)
                            )
                            story.append(
                                Paragraph(
                                    f"Requirements: {hf['note']}", styles["Normal"]
                                )
                            )

                        if "gguf" in install_methods:
                            gguf = install_methods["gguf"]
                            story.append(
                                Paragraph(
                                    f"<b>⚙️ GGUF (llama.cpp):</b>", styles["Normal"]
                                )
                            )
                            story.append(
                                Paragraph(
                                    f"Download: {gguf['source']}", styles["Normal"]
                                )
                            )
                            story.append(
                                Paragraph(
                                    f"Recommended: {gguf['recommended_quant']} | {gguf['note']}",
                                    styles["Normal"],
                                )
                            )

                        story.append(Spacer(1, 12))

        # Add installation platforms guide
        story.append(Paragraph("🛠️ Platform Installation Guide", heading_style))

        platform_data = [
            ["Platform", "Installation Method", "Best For"],
            [
                "Ollama",
                "Download from ollama.ai or use package manager",
                "Beginners, easiest setup",
            ],
            [
                "LM Studio",
                "Download GUI app from lmstudio.ai",
                "Users preferring graphical interface",
            ],
            [
                "llama.cpp",
                "Build from source or download releases",
                "Advanced users, CPU optimization",
            ],
            ["HuggingFace", "pip install transformers torch", "Python developers"],
        ]

        platform_table = Table(
            platform_data, colWidths=[1.5 * inch, 2.5 * inch, 2 * inch]
        )
        platform_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )

        story.append(platform_table)

        # Build PDF
        doc.build(story)
        return output_path

    def _create_html_content(self):
        """Create the complete HTML report content"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Hardware Compatibility Report</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._get_header()}
        {self._get_system_specs()}
        {self._get_recommendations_section()}
        {self._get_footer()}
    </div>
</body>
</html>
"""

    def _get_css_styles(self):
        """Enhanced CSS styles with installation method styling"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .system-specs {
            background-color: #f8f9fa;
        }
        
        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        .specs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .spec-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .spec-label {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .spec-value {
            color: #555;
            font-size: 1.1em;
        }
        
        .model-card {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            transition: transform 0.2s;
        }
        
        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .model-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .model-name {
            font-size: 1.4em;
            font-weight: bold;
            color: #2c3e50;
            margin-right: 15px;
        }
        
        .model-domain {
            background: #e74c3c;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .performance-tier {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .excellent { background-color: #d4edda; color: #155724; }
        .good { background-color: #fff3cd; color: #856404; }
        .basic { background-color: #f8d7da; color: #721c24; }
        
        .requirements {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
            padding: 15px;
            background: white;
            border-radius: 8px;
        }
        
        .req-item {
            text-align: center;
            padding: 10px;
            border-radius: 6px;
            background: #f1f3f4;
        }
        
        .req-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .req-value {
            font-weight: bold;
            color: #333;
        }
        
        .installation-methods {
            margin-top: 20px;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .install-method {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            position: relative;
        }
        
        .install-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .install-title {
            font-weight: bold;
            color: #2c3e50;
            font-size: 1.1em;
        }
        
        .install-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .install-badge.easy { background: #d4edda; color: #155724; }
        .install-badge.intermediate { background: #fff3cd; color: #856404; }
        .install-badge.advanced { background: #f8d7da; color: #721c24; }
        
        .install-command {
            background: #2c3e50;
            color: #fff;
            padding: 12px 16px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            margin: 10px 0;
            position: relative;
            overflow-x: auto;
        }
        
        .install-details {
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .model-id {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .copy-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
        }
        
        .copy-btn:hover {
            background: #0056b3;
        }
        
        .code-block {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            margin: 10px 0;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        
        .install-steps {
            margin: 15px 0;
            color: #555;
        }
        
        .install-steps ol li {
            margin: 5px 0;
        }
        
        .install-note {
            background: #e9ecef;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 8px 0;
            font-size: 0.9em;
            border-left: 4px solid #6c757d;
        }
        
        .source-link {
            color: #007bff;
            text-decoration: none;
            word-break: break-all;
        }
        
        .source-link:hover {
            text-decoration: underline;
        }
        
        .quick-start {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .platform-tabs {
            display: grid;
            gap: 10px;
            margin-top: 10px;
        }
        
        .platform-tab {
            background: white;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #ffd32a;
            font-size: 0.9em;
        }
        
        .platform-tab code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-top: 30px;
            color: #666;
        }
        
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #e9ecef;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        
        @media print {
            body { background: white; }
            .container { box-shadow: none; }
            .model-card { break-inside: avoid; }
            .install-method { break-inside: avoid; }
        }
        
        @media (max-width: 768px) {
            .install-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
            
            .install-details {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .platform-tabs {
                grid-template-columns: 1fr;
            }
        }
        """

    def _create_html_content(self):
        """Create the complete HTML report content"""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LLM Hardware Compatibility Report</title>
            <style>
                {self._get_css_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                {self._get_header()}
                {self._get_system_specs()}
                {self._get_recommendations_section()}
                {self._get_footer()}
            </div>
        </body>
        </html>
        """

    def _get_enhanced_css_styles(self):
        """Enhanced CSS styles with installation method styling"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .system-specs {
            background-color: #f8f9fa;
        }
        
        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        .specs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .spec-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .spec-label {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .spec-value {
            color: #555;
            font-size: 1.1em;
        }
        
        .model-card {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            transition: transform 0.2s;
        }
        
        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .model-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .model-name {
            font-size: 1.4em;
            font-weight: bold;
            color: #2c3e50;
            margin-right: 15px;
        }
        
        .model-domain {
            background: #e74c3c;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .performance-tier {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .excellent { background-color: #d4edda; color: #155724; }
        .good { background-color: #fff3cd; color: #856404; }
        .basic { background-color: #f8d7da; color: #721c24; }
        
        .requirements {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
            padding: 15px;
            background: white;
            border-radius: 8px;
        }
        
        .req-item {
            text-align: center;
            padding: 10px;
            border-radius: 6px;
            background: #f1f3f4;
        }
        
        .req-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .req-value {
            font-weight: bold;
            color: #333;
        }
        
        .installation-methods {
            margin-top: 20px;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .install-method {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            position: relative;
        }
        
        .install-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .install-title {
            font-weight: bold;
            color: #2c3e50;
            font-size: 1.1em;
        }
        
        .install-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .install-badge.easy { background: #d4edda; color: #155724; }
        .install-badge.intermediate { background: #fff3cd; color: #856404; }
        .install-badge.advanced { background: #f8d7da; color: #721c24; }
        
        .install-command {
            background: #2c3e50;
            color: #fff;
            padding: 12px 16px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            margin: 10px 0;
            position: relative;
            overflow-x: auto;
        }
        
        .install-details {
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .model-id {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .copy-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
        }
        
        .copy-btn:hover {
            background: #0056b3;
        }
        
        .code-block {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            margin: 10px 0;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        
        .install-steps {
            margin: 15px 0;
            color: #555;
        }
        
        .install-steps ol li {
            margin: 5px 0;
        }
        
        .install-note {
            background: #e9ecef;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 8px 0;
            font-size: 0.9em;
            border-left: 4px solid #6c757d;
        }
        
        .source-link {
            color: #007bff;
            text-decoration: none;
            word-break: break-all;
        }
        
        .source-link:hover {
            text-decoration: underline;
        }
        
        .quick-start {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .platform-tabs {
            display: grid;
            gap: 10px;
            margin-top: 10px;
        }
        
        .platform-tab {
            background: white;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #ffd32a;
            font-size: 0.9em;
        }
        
        .platform-tab code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-top: 30px;
            color: #666;
        }
        
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #e9ecef;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        
        @media print {
            body { background: white; }
            .container { box-shadow: none; }
            .model-card { break-inside: avoid; }
            .install-method { break-inside: avoid; }
        }
        
        @media (max-width: 768px) {
            .install-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
            
            .install-details {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .platform-tabs {
                grid-template-columns: 1fr;
            }
        }
        """

    def _get_header(self):
        """Generate HTML header section"""
        return f"""
        <div class="header">
            <h1>🤖 LLM Hardware Compatibility Report</h1>
            <div class="subtitle">Generated on {self.timestamp.strftime('%B %d, %Y at %I:%M %p')}</div>
        </div>
        """

    def _get_system_specs(self):
        """Generate system specifications section"""
        specs_html = f"""
        <div class="section system-specs">
            <h2>🖥️ System Specifications</h2>
            <div class="specs-grid">
                <div class="spec-item">
                    <div class="spec-label">Operating System</div>
                    <div class="spec-value">{self.system_info['os']} ({self.system_info['architecture']})</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Processor</div>
                    <div class="spec-value">{self.system_info['processor']}</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">CPU Cores</div>
                    <div class="spec-value">{self.system_info['cpu_cores']} physical / {self.system_info['cpu_threads']} logical</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Memory (RAM)</div>
                    <div class="spec-value">{self.system_info['total_ram_gb']} GB total ({self.system_info['available_ram_gb']} GB available)</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Storage</div>
                    <div class="spec-value">{self.system_info['free_storage_gb']} GB free / {self.system_info['total_storage_gb']} GB total</div>
                </div>
        """

        if self.system_info["gpus"]:
            for gpu in self.system_info["gpus"]:
                vram_info = (
                    f"{gpu['vram_gb']} GB VRAM"
                    if isinstance(gpu["vram_gb"], (int, float))
                    else gpu["vram_gb"]
                )
                specs_html += f"""
                <div class="spec-item">
                    <div class="spec-label">GPU</div>
                    <div class="spec-value">{gpu['name']} ({vram_info})</div>
                </div>
                """
        else:
            specs_html += """
                <div class="spec-item">
                    <div class="spec-label">GPU</div>
                    <div class="spec-value">None detected</div>
                </div>
            """

        specs_html += """
            </div>
        </div>
        """

        return specs_html

    def _get_recommendations_section(self):
        """Generate recommendations section"""
        suitable_models = (
            self.recommendations["excellent"]
            + self.recommendations["good"]
            + self.recommendations["basic"]
        )

        if not suitable_models:
            return self._get_insufficient_hardware_html()

        # Generate summary statistics
        total_suitable = len(suitable_models)
        total_specialized = sum(
            len([m for m in self.recommendations[cat] if "domain" in m["specs"]])
            for cat in ["excellent", "good", "basic"]
        )
        total_general = total_suitable - total_specialized

        html = f"""
        <div class="section">
            <h2>🤖 Model Recommendations</h2>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-number">{total_suitable}</div>
                    <div class="stat-label">Compatible Models</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_general}</div>
                    <div class="stat-label">General Purpose</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_specialized}</div>
                    <div class="stat-label">Specialized Domain</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(self.recommendations['not_suitable'])}</div>
                    <div class="stat-label">Not Suitable</div>
                </div>
            </div>
        """

        # Add model recommendations
        categories = [
            (
                "excellent",
                "🟢 Excellent Performance",
                self.recommendations["excellent"],
            ),
            ("good", "🟡 Good Performance", self.recommendations["good"]),
            ("basic", "🟠 Basic Performance", self.recommendations["basic"]),
        ]

        for category, title, models in categories:
            if models:
                html += f"<h3>{title}</h3>"
                for model in models:
                    html += self._generate_model_card_html(model, category)

        html += "</div>"
        return html

    """
Enhanced Report Generation Methods with Complete Installation Instructions
Add these methods to replace the existing ones in the ReportGenerator class
"""

    def _generate_model_card_html(self, model, category):
        """Generate HTML for a single model card with complete installation instructions"""
        model_name = model["name"]
        specs = model["specs"]
        compatibility = model["compatibility"]

        domain_badge = ""
        if "domain" in specs:
            domain_badge = f'<span class="model-domain">{specs["domain"]}</span>'

        performance_class = category

        card_html = f"""
        <div class="model-card">
            <div class="model-header">
                <div class="model-name">{model_name}</div>
                {domain_badge}
            </div>
            
            <div class="performance-tier {performance_class}">
                {compatibility['performance_tier']}
            </div>
            
            <p style="margin: 15px 0; color: #555;">{specs['description']}</p>
            
            <div class="requirements">
                <div class="req-item">
                    <div class="req-label">Parameters</div>
                    <div class="req-value">{specs['parameters']}</div>
                </div>
                <div class="req-item">
                    <div class="req-label">Min RAM</div>
                    <div class="req-value">{specs['min_ram_gb']} GB</div>
                </div>
                <div class="req-item">
                    <div class="req-label">Rec. RAM</div>
                    <div class="req-value">{specs['recommended_ram_gb']} GB</div>
                </div>
                <div class="req-item">
                    <div class="req-label">Min VRAM</div>
                    <div class="req-value">{specs['min_vram_gb']} GB</div>
                </div>
                <div class="req-item">
                    <div class="req-label">Rec. VRAM</div>
                    <div class="req-value">{specs['recommended_vram_gb']} GB</div>
                </div>
            </div>
        """

        if compatibility["recommended_quant"]:
            card_html += f'<p><strong>🔧 Recommended Quantization:</strong> {compatibility["recommended_quant"]}</p>'

        if compatibility["notes"]:
            notes = "; ".join(compatibility["notes"])
            card_html += f"<p><strong>ℹ️ Notes:</strong> {notes}</p>"

        # Enhanced Installation Methods Section
        card_html += """
        <div class="installation-methods">
            <h4 style="color: #2c3e50; margin: 20px 0 15px 0; font-size: 1.1em;">🚀 Installation & Setup Instructions</h4>
        """

        install_methods = specs["install_methods"]

        if "ollama" in install_methods:
            ollama = install_methods["ollama"]
            card_html += f"""
            <div class="install-method">
                <div class="install-header">
                    <div class="install-title">📱 OLLAMA (Recommended for Beginners)</div>
                    <div class="install-badge easy">Easy Setup</div>
                </div>
                <div class="install-command">{ollama['command']}</div>
                <div class="install-steps">
                    <strong>Setup Steps:</strong>
                    <ol style="margin: 8px 0 0 20px; line-height: 1.6;">
                        <li>Install Ollama from <a href="https://ollama.ai" target="_blank">ollama.ai</a></li>
                        <li>Open terminal/command prompt</li>
                        <li>Run the command above</li>
                        <li>Model will auto-download and start</li>
                    </ol>
                </div>
                <div class="install-note">✅ <strong>Why? </strong> {ollama['note']}</div>
            </div>
            """

        if "huggingface" in install_methods:
            hf = install_methods["huggingface"]
            card_html += f"""
            <div class="install-method">
                <div class="install-header">
                    <div class="install-title">🤗 HUGGING FACE (For Developers)</div>
                    <div class="install-badge intermediate">Intermediate</div>
                </div>
                <div class="install-details">
                    <strong>Model ID:</strong> <code class="model-id">{hf['model_id']}</code>
                    <button class="copy-btn" onclick="navigator.clipboard.writeText('{hf['model_id']}')">📋 Copy</button>
                </div>
                <div class="install-steps">
                    <strong>Python Setup:</strong>
                    <div class="code-block">
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Load model and tokenizer
                model = AutoModelForCausalLM.from_pretrained("{hf['model_id']}")
                tokenizer = AutoTokenizer.from_pretrained("{hf['model_id']}")

                # Generate text
                inputs = tokenizer("Your prompt here", return_tensors="pt")
                outputs = model.generate(**inputs, max_length=100)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(response)
                    </div>
                </div>
                <div class="install-note">💡 <strong>Requirements:</strong> {hf['note']}</div>
                <div class="install-note">🔧 <strong>Install:</strong> pip install transformers torch</div>
            </div>
            """

        if "gguf" in install_methods:
            gguf = install_methods["gguf"]
            card_html += f"""
            <div class="install-method">
                <div class="install-header">
                    <div class="install-title">⚙️ GGUF (Advanced Users)</div>
                    <div class="install-badge advanced">Advanced</div>
                </div>
                <div class="install-details">
                    <strong>Download Source:</strong> 
                    <a href="{gguf['source']}" target="_blank" class="source-link">{gguf['source']}</a>
                </div>
                <div class="install-steps">
                    <strong>llama.cpp Setup:</strong>
                    <ol style="margin: 8px 0 0 20px; line-height: 1.6;">
                        <li>Install llama.cpp from <a href="https://github.com/ggerganov/llama.cpp" target="_blank">GitHub</a></li>
                        <li>Download the {gguf['recommended_quant']} quantized model</li>
                        <li>Run: <code>./main -m model.gguf -p "Your prompt"</code></li>
                    </ol>
                </div>
                <div class="install-note">🎯 <strong>Recommended:</strong> {gguf['recommended_quant']}</div>
                <div class="install-note">💡 <strong>Best for:</strong> {gguf['note']}</div>
            </div>
            """

        # Add platform-specific quick start
        card_html += f"""
        <div class="quick-start">
            <h5 style="color: #2c3e50; margin: 15px 0 10px 0;">⚡ Quick Start Guide</h5>
            <div class="platform-tabs">
                <div class="platform-tab">
                    <strong>🪟 Windows:</strong> 
                    Download Ollama installer → Run → <code>{install_methods.get('ollama', {}).get('command', 'ollama run model')}</code>
                </div>
                <div class="platform-tab">
                    <strong>🍎 macOS:</strong> 
                    <code>brew install ollama</code> → <code>{install_methods.get('ollama', {}).get('command', 'ollama run model')}</code>
                </div>
                <div class="platform-tab">
                    <strong>🐧 Linux:</strong> 
                    <code>curl -fsSL https://ollama.ai/install.sh | sh</code> → <code>{install_methods.get('ollama', {}).get('command', 'ollama run model')}</code>
                </div>
            </div>
        </div>
        """

        card_html += "</div></div>"
        return card_html

    def _get_insufficient_hardware_html(self):
        """Generate HTML for insufficient hardware scenario"""
        ram_gb = self.system_info["total_ram_gb"]
        storage_gb = self.system_info["free_storage_gb"]

        return f"""
        <div class="section">
            <h2>❌ Insufficient Hardware Detected</h2>
            
            <div style="background: #f8d7da; color: #721c24; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3>Your Current System:</h3>
                <ul style="margin: 10px 0 0 20px;">
                    <li>RAM: {ram_gb} GB</li>
                    <li>Free Storage: {storage_gb} GB</li>
                    <li>GPUs: {len(self.system_info['gpus'])} detected</li>
                </ul>
                
                <p style="margin-top: 15px;">
                    Unfortunately, your system doesn't meet the minimum requirements 
                    for running local LLMs efficiently.
                </p>
            </div>
            
            <h3>🌐 Recommended Cloud-Based Solutions:</h3>
            <ul style="margin: 15px 0 0 20px; line-height: 1.8;">
                <li><strong>ChatGPT:</strong> https://chat.openai.com</li>
                <li><strong>Claude:</strong> https://claude.ai</li>
                <li><strong>Google Bard:</strong> https://bard.google.com</li>
                <li><strong>Perplexity AI:</strong> https://perplexity.ai</li>
            </ul>
        </div>
        """

    def _get_footer(self):
        """Generate HTML footer"""
        return f"""
            <div class="footer">
                <p>Generated by LLM Hardware Compatibility Checker</p>
                <p>Report created on {self.timestamp.strftime('%B %d, %Y at %I:%M:%S %p')}</p>
            </div>
            """

def install_requirements():
    """Install required packages if missing"""
    required_packages = ["psutil"]
    optional_packages = ["GPUtil", "nvidia-ml-py3", "reportlab"]

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
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_optional.append(package)

    if missing_optional:
        print(f"⚠️  Optional packages not found: {missing_optional}")
        print(
            "For enhanced features, install with: pip install "
            + " ".join(missing_optional)
        )
        print("   • GPUtil/nvidia-ml-py3: Better GPU detection")
        print("   • reportlab: PDF report generation")

    return True
    
def main():
    """Main function with report generation options"""
    parser = argparse.ArgumentParser(
        description="LLM Hardware Compatibility Checker"
    )
    parser.add_argument(
        "--report",
        choices=["html", "pdf", "both"],
        help="Generate report in specified format(s)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save reports (default: current directory)",
    )

    args = parser.parse_args()

    print("🚀 LLM Hardware Compatibility Checker & Installation Guide")
    print("=" * 60)

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

    # Generate reports if requested
    if args.report:
        print(f"\n🔄 Generating {args.report} report(s)...")
        report_results = recommender.generate_report(args.report, args.output_dir)

        print("\n📄 Report Generation Complete!")
        if "html" in report_results and report_results["html"]:
            print(f"   📄 HTML: {report_results['html']}")
        if "pdf" in report_results and report_results["pdf"]:
            print(f"   📄 PDF: {report_results['pdf']}")

    # Check if any models are suitable for final message
    suitable_models = (
        recommendations["excellent"]
        + recommendations["good"]
        + recommendations["basic"]
    )

    print("\n" + "=" * 60)
    if suitable_models:
        print("✅ Analysis complete! Follow the installation instructions above.")
        print("💡 Tip: Start with Ollama if you're new to running local LLMs!")
    else:
        print("📊 Analysis complete! Check the cloud-based alternatives above.")
        print(
            "💡 Tip: Cloud LLM services often perform better than small local models!"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
