# 🤖 LLM Hardware Compatibility Checker

A comprehensive Python script that analyzes your system hardware and provides personalized recommendations for running Large Language Models (LLMs) locally, complete with specific installation instructions and download links.

## 🎯 Overview

This script solves the common problem: **"Can my computer run local LLMs, and if so, which ones?"**

### What it does:

- 🔍 **Analyzes your hardware**: CPU, RAM, GPU (NVIDIA/AMD/Apple Silicon), and storage
- 🎯 **Recommends compatible LLMs**: From 2B to 70B parameter models
- 📦 **Provides installation instructions**: Specific commands for Ollama, HuggingFace, llama.cpp, and more
- ⚡ **Suggests optimizations**: Quantization levels, platform-specific tips
- 🌐 **Offers alternatives**: Cloud services when local hardware is insufficient
- 🛠️ **Cross-platform support**: Works on Windows, macOS, and Linux

### Supported LLM Sources:
- **Ollama** (Recommended for beginners)
- **HuggingFace Transformers**
- **GGUF Models** (llama.cpp compatible)
- **LM Studio**

## 🚀 Installation

### Minimum System Requirements:
- **Python**: 3.7 or higher
- **RAM**: 1GB for the script itself
- **Storage**: 100MB for dependencies
- **OS**: Windows 10+, macOS 10.14+, or Linux

### Quick Install:

```bash
# Clone the repository
git clone https://github.com/your-username/llm-hardware-checker.git
cd llm-hardware-checker

# Install required dependencies
pip install psutil

# Optional: For better GPU detection
pip install GPUtil nvidia-ml-py3
```

### Dependencies:

**Required:**
- `psutil` - System information gathering

**Optional (for enhanced GPU detection):**
- `GPUtil` - NVIDIA GPU detection
- `nvidia-ml-py3` - Advanced NVIDIA GPU monitoring

If optional packages aren't installed, the script will still work but with limited GPU detection capabilities.

## 📝 How to Run

### Basic Usage:

```bash
python llm_checker.py
```

### With Virtual Environment (Recommended):

```bash
# Create virtual environment
python -m venv llm-checker-env

# Activate it
# On Windows:
llm-checker-env\Scripts\activate
# On macOS/Linux:
source llm-checker-env/bin/activate

# Install dependencies and run
pip install psutil
python llm_checker.py
```

The script will automatically:
1. Detect your hardware specifications
2. Analyze compatibility with popular LLMs
3. Provide specific installation commands
4. Suggest optimizations for your system

## ✅ Example Output: Compatible Hardware

Here's what you'll see on a system that can run LLMs:

```
🚀 LLM Hardware Compatibility Checker & Installation Guide
============================================================

🔍 Analyzing your system hardware...

==================================================
🖥️  SYSTEM SPECIFICATIONS
==================================================
OS: Linux (x86_64)
CPU: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
CPU Cores: 8 physical / 16 logical
RAM: 32.0 GB total (28.5 GB available)
Storage: 450.2 GB free / 1000.0 GB total
GPUs:
  - NVIDIA GeForce RTX 3070 (8.0 GB VRAM)

==================================================
🤖 LLM RECOMMENDATIONS & INSTALLATION GUIDE
==================================================

🟢 EXCELLENT PERFORMANCE
------------------------

1. 📦 Llama 3.1 8B
   📊 8B parameters
   📝 Excellent balance of performance and efficiency
   💾 RAM: 8GB min / 16GB recommended
   🎮 VRAM: 5GB min / 8GB recommended
   ⚡ Status: Excellent (GPU)
   🔧 Recommended quantization: Q8_0 or FP16

   🚀 INSTALLATION OPTIONS:

   📱 OLLAMA (Recommended for beginners):
      Command: ollama run llama3.1:8b
      Note: Most popular choice

   🤗 HUGGING FACE:
      Model ID: meta-llama/Meta-Llama-3.1-8B-Instruct
      Usage: Requires HF login
      Note: Full precision model

   ⚙️  GGUF (llama.cpp compatible):
      Source: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
      Recommended: Q4_K_M for 8GB VRAM, Q8_0 for 16GB+
      Note: Choose quantization based on your VRAM

🟡 GOOD PERFORMANCE
-------------------

2. 📦 Mistral 7B
   📊 7B parameters
   📝 High-quality model with strong performance
   💾 RAM: 6GB min / 12GB recommended
   🎮 VRAM: 4GB min / 6GB recommended
   ⚡ Status: Good (GPU)
   🔧 Recommended quantization: Q4_K_M

   🚀 INSTALLATION OPTIONS:
   [... installation details ...]

==================================================
🛠️  INSTALLATION PLATFORMS
==================================================

1️⃣  OLLAMA - Easiest Option (Recommended)
   📥 Installation:
      • Run: curl -fsSL https://ollama.ai/install.sh | sh
      • Or download from https://ollama.ai/download/linux
      • Open terminal
   🚀 Usage:
      • ollama run <model-name>
      • Example: ollama run llama3.1:8b
      • Models auto-download on first use

[... more platform details ...]

==================================================
💡 OPTIMIZATION TIPS FOR YOUR SYSTEM
==================================================

🔧 HIGH RAM OPTIMIZATION:
   • Can use Q5_K_M or Q8_0 for better quality
   • Multiple models can be loaded simultaneously

🎮 GPU OPTIMIZATION (Good VRAM):
   • Q4_K_M or Q5_K_M work well
   • 7B models will run smoothly

============================================================
✅ Analysis complete! Follow the installation instructions above.
💡 Tip: Start with Ollama if you're new to running local LLMs!
============================================================
```

## ❌ Example Output: Insufficient Hardware

Here's what users see when their hardware cannot run local LLMs:

```
🚀 LLM Hardware Compatibility Checker & Installation Guide
============================================================

🔍 Analyzing your system hardware...

==================================================
🖥️  SYSTEM SPECIFICATIONS
==================================================
OS: Windows (AMD64)
CPU: Intel(R) Core(TM) i3-6100 CPU @ 3.70GHz
CPU Cores: 2 physical / 4 logical
RAM: 4.0 GB total (2.8 GB available)
Storage: 12.5 GB free / 120.0 GB total
GPUs: None detected

==================================================
🤖 LLM RECOMMENDATIONS & INSTALLATION GUIDE
==================================================

❌ INSUFFICIENT HARDWARE DETECTED
==================================================
📊 Your current system:
   • RAM: 4.0 GB
   • Free Storage: 12.5 GB
   • GPUs: 0 detected

⚠️  Unfortunately, your system doesn't meet the minimum requirements
   for running local LLMs efficiently.

🔍 Specific Issues:
   • Insufficient RAM: 4.0 GB (minimum 6 GB needed for smallest models)
   • Low storage space: 12.5 GB (minimum 20 GB needed)

💡 RECOMMENDED SOLUTIONS:
==============================

1️⃣  CLOUD-BASED LLM SERVICES (Recommended)
   🌐 Use online LLM services instead:
   • ChatGPT (https://chat.openai.com)
   • Claude (https://claude.ai)
   • Google Bard (https://bard.google.com)
   • Perplexity AI (https://perplexity.ai)
   • Hugging Face Spaces (https://huggingface.co/spaces)
   ✅ Pros: No hardware requirements, always up-to-date
   ❌ Cons: Requires internet, may have usage limits

2️⃣  CLOUD COMPUTING PLATFORMS
   ☁️  Rent powerful hardware temporarily:
   • Google Colab (Free tier available)
   • Kaggle Notebooks (Free GPU hours)
   • AWS EC2 with GPU instances
   • Google Cloud Platform
   • RunPod (GPU rentals)

3️⃣  HARDWARE UPGRADE OPTIONS
   🔧 Minimum recommended upgrades:
   • RAM: Upgrade to at least 8 GB (16 GB preferred)
     Current: 4.0 GB → Target: 8-16 GB
   • Storage: Free up space or add storage
     Current: 12.5 GB free → Target: 50+ GB free

📏 SMALLEST MODEL REQUIREMENTS:
   Model: Gemma 2B
   Minimum RAM: 6.0 GB
   Your RAM: 4.0 GB
   Gap: 2.0 GB short

🎯 IMMEDIATE NEXT STEPS:
   1. Try cloud-based LLM services (free to start)
   2. Check if you can free up RAM by closing other programs
   3. Consider Google Colab for free GPU access
   4. Plan hardware upgrades if you want local LLMs

💬 Don't worry! You have many options to work with LLMs.
   Cloud services often provide better performance than
   running smaller models locally anyway!

============================================================
📊 Analysis complete! Check the cloud-based alternatives above.
💡 Tip: Cloud LLM services often perform better than small local models!
============================================================
```

## 🔧 Supported Models

The script analyzes compatibility with **popular, locally-deployable LLMs**. This represents a curated selection of the most practical models for local deployment:

| Model | Parameters | Min RAM | Min VRAM | Best For |
|-------|------------|---------|----------|----------|
| Gemma 2B | 2B | 3GB | 1.5GB | Low-resource systems |
| Phi-3 Mini | 3.8B | 4GB | 2GB | General tasks |
| Llama 3.2 3B | 3B | 4GB | 2GB | Balanced performance |
| Mistral 7B | 7B | 6GB | 4GB | High quality responses |
| Llama 3.1 8B | 8B | 8GB | 5GB | Most popular choice |
| Llama 3.1 13B | 13B | 12GB | 8GB | Advanced tasks |
| Code Llama 7B | 7B | 6GB | 4GB | Programming assistance |
| Llama 3.1 70B | 70B | 48GB | 40GB | State-of-the-art performance |

### 📋 Coverage Notes

**✅ Included:**
- Most popular open-source models for local deployment
- Models with excellent community support and tooling
- Representative range from 2B to 70B parameters

**❌ Not Included:**
- **API-only models**: GPT-4, Claude, Gemini (can't run locally)
- **Specialized models**: Medical, legal, finance-specific LLMs
- **Experimental models**: Research-only or unstable releases
- **All parameter variants**: Every possible size configuration

**🔍 Finding More Models:**
The script includes guidance on finding additional models from:
- HuggingFace Model Hub
- Ollama Library  
- Specialized model repositories

For a complete list of all available LLMs, visit [HuggingFace Models](https://huggingface.co/models?pipeline_tag=text-generation).

## 🌟 Features

- **✅ Accurate Hardware Detection**: Detects CPU, RAM, GPU (NVIDIA/AMD/Apple Silicon), and storage
- **🎯 Smart Recommendations**: Matches your hardware to compatible models
- **📦 Installation Ready**: Provides exact commands to install and run models
- **⚡ Performance Optimization**: Suggests optimal quantization and settings
- **🌐 Fallback Options**: Cloud alternatives when local hardware isn't sufficient
- **🔄 Cross-Platform**: Works on Windows, macOS, and Linux
- **📊 Detailed Analysis**: Shows performance tiers and resource requirements
- **🛠️ Multiple Platforms**: Supports Ollama, HuggingFace, llama.cpp, and more

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

### 🔧 Adding New Models
To add a new model to the database, update the `create_llm_database()` method with:
```python
"Model Name": {
    "parameters": "7B",
    "min_ram_gb": 6,
    "recommended_ram_gb": 12,
    "min_vram_gb": 4,
    "recommended_vram_gb": 6,
    "cpu_only": True,
    "description": "Model description",
    "install_methods": {
        "ollama": {"command": "ollama run model", "note": "Installation note"},
        "huggingface": {"model_id": "org/model-name", "command": "Usage info", "note": "Notes"},
        "gguf": {"source": "https://...", "recommended_quant": "Q4_K_M", "note": "Notes"}
    }
}
```

### 🎯 Priority Areas
- **New model families**: Mixtral, Falcon, Yi, Qwen, etc.
- **Specialized models**: Medical, legal, scientific LLMs
- **Better GPU detection**: AMD/Intel GPU support
- **Mobile deployment**: Support for mobile/edge devices
- **Cloud integration**: Better cloud platform recommendations

### 📝 Guidelines
- Test models on real hardware before adding
- Include accurate memory requirements
- Provide working installation commands
- Update documentation and tests

Please [open an issue](https://github.com/your-username/llm-hardware-checker/issues) to discuss new features before submitting PRs.

## ❓ Frequently Asked Questions

### Q: Does this script cover ALL LLMs?
**A: No** - it covers the most popular and practical models for local deployment. Here's why:

- **Focus on local deployment**: Excludes API-only models (GPT-4, Claude, Gemini)
- **Quality over quantity**: Curated selection of well-supported, stable models
- **Maintenance**: Adding every model would be impractical as new ones release daily
- **Practical use**: Covers 90% of what people actually run locally

For additional models, the script provides guidance on finding them from HuggingFace, Ollama, and other repositories.

### Q: Why isn't [specific model] included?
**A:** Models are selected based on:
- **Popularity and adoption**
- **Local deployment feasibility** 
- **Community support and tooling**
- **Stability and reliability**

If there's a model you think should be included, please [open an issue](https://github.com/your-username/llm-hardware-checker/issues) with details.

### Q: Can I add my own models?
**A:** Yes! The script is designed to be extensible. See the Contributing section for guidelines on adding new models.

### Q: What about commercial models?
**A:** Commercial/proprietary models that require API access (GPT-4, Claude, etc.) aren't suitable for local deployment analysis. The script focuses on open-source models you can download and run yourself.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Support

If this script helped you find the right LLM for your system, please consider:

⭐ **Starring this repository** - it helps others discover this tool!

Found a bug or have a feature request? Please [open an issue](https://github.com/your-username/llm-hardware-checker/issues).

---

**Made with ❤️ for the AI community. Happy LLM running! 🤖**

---

### 📊 Script Statistics
- **Models Analyzed**: 12+ popular LLMs
- **Platforms Supported**: 4 installation methods
- **Hardware Types**: CPU, GPU (NVIDIA/AMD/Apple), RAM, Storage
- **Operating Systems**: Windows, macOS, Linux