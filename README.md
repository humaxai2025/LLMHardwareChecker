# 🤖 LLM Hardware Compatibility Checker

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform Support](https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-lightgrey)](https://github.com/humaxai2025/LLMHardwareChecker)

A comprehensive Python script that analyzes your system hardware and provides personalized recommendations for running Large Language Models (LLMs) locally, including both general-purpose and specialized domain models, complete with specific installation instructions, download links, and professional report generation.

## 🎯 Overview

This script solves the common problem: **"Can my computer run local LLMs, and if so, which ones?"**

### What it does:

- 🔍 **Analyzes your hardware**: CPU, RAM, GPU (NVIDIA/AMD/Apple Silicon), and storage
- 🎯 **Recommends compatible LLMs**: From 2B to 70B parameter models
- 🔬 **Includes specialized models**: Domain-specific models for coding, medical, math, and research
- 📦 **Provides installation instructions**: Specific commands for Ollama, HuggingFace, llama.cpp, and more
- 📄 **Generates professional reports**: Beautiful HTML and PDF reports for documentation
- ⚡ **Suggests optimizations**: Quantization levels, platform-specific tips
- 🌐 **Offers alternatives**: Cloud services when local hardware is insufficient
- 🛠️ **Cross-platform support**: Works on Windows, macOS, and Linux

### Model Categories:
- **🌟 General Purpose**: Llama, Mistral, Gemma, Phi models for everyday use
- **🔬 Specialized Domains**: 
  - **Code Generation**: StarCoder, WizardCoder, Code Llama
  - **Medical/Biology**: BioMistral  
  - **Mathematics**: MetaMath
  - **Research/Analysis**: Nous Hermes 2

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
git clone https://github.com/humaxai2025/LLMHardwareChecker.git
cd LLMHardwareChecker

# Install required dependencies
pip install psutil

# Optional: For enhanced features
pip install reportlab GPUtil nvidia-ml-py3
```

### Dependencies:

**Required:**
- `psutil` - System information gathering

**Optional (for enhanced features):**
- `reportlab` - Professional PDF report generation
- `GPUtil` - NVIDIA GPU detection
- `nvidia-ml-py3` - Advanced NVIDIA GPU monitoring

If optional packages aren't installed, the script will still work but with limited GPU detection and no PDF reports.

## 📝 How to Run

### Basic Usage:

```bash
python llmhardwarechecker.py
```

### With Report Generation:

```bash
# Generate HTML report only
python llmhardwarechecker.py --report html

# Generate PDF report only  
python llmhardwarechecker.py --report pdf

# Generate both HTML and PDF reports
python llmhardwarechecker.py --report both

# Save reports to specific directory
python llmhardwarechecker.py --report both --output-dir ./reports
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
pip install psutil reportlab
python llmhardwarechecker.py --report both
```

The script will automatically:
1. Detect your hardware specifications
2. Analyze compatibility with popular LLMs
3. Provide specific installation commands
4. Suggest optimizations for your system
5. Generate professional reports (if requested)

## 🔧 Supported Models

The script analyzes compatibility with **popular, locally-deployable LLMs** including both general-purpose and specialized domain models:

### 🌟 General Purpose Models

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

### 🔬 Specialized Domain Models

| Model | Domain | Parameters | Min RAM | Min VRAM | Best For |
|-------|--------|------------|---------|----------|----------|
| StarCoder 7B | Code Generation | 7B | 6GB | 4GB | Multi-language coding (80+ languages) |
| WizardCoder 7B | Code Generation | 7B | 6GB | 4GB | Advanced coding with reasoning |
| BioMistral 7B | Medical/Biology | 7B | 6GB | 4GB | Healthcare & medical research |
| MetaMath 7B | Mathematics | 7B | 6GB | 4GB | Mathematical problem solving |
| Nous Hermes 2 Solar | Research/Analysis | 10.7B | 8GB | 6GB | Research & analytical tasks |

### 📋 Coverage Notes

**✅ Included:**
- **General models**: Most popular open-source models for everyday use
- **Specialized models**: Domain-specific models that are truly deployable
- **Range of sizes**: From 2B to 70B parameters for different hardware capabilities
- **Proven track record**: Established models with excellent community support

**🔬 Domain Specializations:**
- **Code Generation**: Multi-language programming, debugging, code analysis
- **Medical/Biology**: Healthcare Q&A, medical research, biological text analysis  
- **Mathematics**: Step-by-step problem solving, mathematical reasoning
- **Research/Analysis**: Academic research, data analysis, complex reasoning

**❌ Not Included:**
- **API-only models**: GPT-4, Claude, Gemini (can't run locally)
- **Experimental models**: Research-only or unstable releases
- **Commercial-only models**: Models requiring paid licenses for local deployment

**🔍 Finding More Models:**
The script includes guidance on finding additional models from:
- HuggingFace Model Hub (17,000+ text generation models)
- Ollama Library (curated local deployment models)
- Specialized model repositories

For a complete list of all available LLMs, visit [HuggingFace Models](https://huggingface.co/models?pipeline_tag=text-generation).

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

🌟 GENERAL PURPOSE MODELS
========================================

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

🔬 SPECIALIZED DOMAIN MODELS
========================================
These models are optimized for specific use cases and domains:

🟢 EXCELLENT PERFORMANCE
------------------------

2. 📦 StarCoder 7B
   🎯 Domain: Code Generation
   📊 7B parameters
   📝 Multi-language code generation, supports 80+ programming languages
   💾 RAM: 6GB min / 12GB recommended
   🎮 VRAM: 4GB min / 6GB recommended
   ⚡ Status: Excellent (GPU)

   🚀 INSTALLATION OPTIONS:

   📱 OLLAMA (Recommended for beginners):
      Command: ollama run starcoder:7b
      Note: Excellent for diverse programming languages

🔄 Generating both report(s)...
📄 HTML report generated: llm_compatibility_report_20241220_143052.html
📄 PDF report generated: llm_compatibility_report_20241220_143052.pdf

📄 Report Generation Complete!
   📄 HTML: llm_compatibility_report_20241220_143052.html
   📄 PDF: llm_compatibility_report_20241220_143052.pdf

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

## 📄 Report Generation Features

### **HTML Reports:**
- **Professional styling** with modern CSS and responsive design
- **Interactive elements** with hover effects and smooth animations
- **Color-coded performance tiers** (Green/Yellow/Orange/Red)
- **Grid layouts** for system specifications and model cards
- **Print-friendly** formatting for physical documentation
- **Summary statistics** with visual cards showing compatible models

### **PDF Reports:**
- **Professional tables** with proper formatting and spacing
- **System specifications** in clean tabular format
- **Model recommendations** with detailed descriptions and requirements
- **Installation commands** included for each compatible model
- **Suitable for sharing** with colleagues, clients, or documentation
- **Consistent layout** across all platforms and devices

### **Report Content:**
- Complete system hardware analysis
- Compatible model recommendations with performance tiers
- Installation instructions for multiple platforms
- Optimization tips specific to your hardware
- Domain-specific model categorization
- Professional formatting suitable for business use

## 🌟 Features

- **✅ Accurate Hardware Detection**: Detects CPU, RAM, GPU (NVIDIA/AMD/Apple Silicon), and storage
- **🎯 Smart Recommendations**: Matches your hardware to compatible models
- **🔬 Specialized Domain Models**: Includes domain-specific models for coding, medical, math, and research
- **📦 Installation Ready**: Provides exact commands to install and run models
- **📄 Professional Reports**: Generate beautiful HTML and PDF documentation
- **⚡ Performance Optimization**: Suggests optimal quantization and settings
- **🌐 Fallback Options**: Cloud alternatives when local hardware isn't sufficient
- **🔄 Cross-Platform**: Works on Windows, macOS, and Linux
- **📊 Detailed Analysis**: Shows performance tiers and resource requirements
- **🛠️ Multiple Platforms**: Supports Ollama, HuggingFace, llama.cpp, and more
- **🎯 Domain Guidance**: Clear categorization between general and specialized models

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
    "domain": "Code Generation",  # For specialized models
    "install_methods": {
        "ollama": {"command": "ollama run model", "note": "Installation note"},
        "huggingface": {"model_id": "org/model-name", "command": "Usage info", "note": "Notes"},
        "gguf": {"source": "https://...", "recommended_quant": "Q4_K_M", "note": "Notes"}
    }
}
```

### 🎯 Priority Areas
- **New model families**: Mixtral, Falcon, Yi, Qwen, etc.
- **More specialized models**: Legal (LawGPT), Finance (FinGPT), Scientific (SciBERT)
- **Better GPU detection**: AMD/Intel GPU support
- **Mobile deployment**: Support for mobile/edge devices
- **Cloud integration**: Better cloud platform recommendations
- **Domain expansion**: Additional specialized domains based on user demand
- **Report enhancements**: Additional report formats and customization options

### 📝 Guidelines
- Test models on real hardware before adding
- Include accurate memory requirements
- Provide working installation commands
- Update documentation and tests
- Ensure specialized models are truly deployable locally

Please [open an issue](https://github.com/humaxai2025/LLMHardwareChecker/issues) to discuss new features before submitting PRs.

## ❓ Frequently Asked Questions

### Q: Does this script cover ALL LLMs?
**A: No** - it covers the most popular and practical models for local deployment, plus selected specialized domain models. Here's why:

- **Focus on local deployment**: Excludes API-only models (GPT-4, Claude, Gemini)
- **Quality over quantity**: Curated selection of well-supported, stable models
- **Domain coverage**: Includes specialized models for common professional domains
- **Maintenance**: Adding every model would be impractical as new ones release daily
- **Practical use**: Covers 90%+ of what people actually run locally

For additional models, the script provides guidance on finding them from HuggingFace, Ollama, and other repositories.

### Q: What specialized domains are covered?
**A:** Currently includes:
- **Code Generation**: StarCoder, WizardCoder, Code Llama (multiple variants)
- **Medical/Biology**: BioMistral for healthcare professionals  
- **Mathematics**: MetaMath for problem-solving
- **Research/Analysis**: Nous Hermes 2 for analytical work

More domains (Legal, Finance, Scientific) are being evaluated for future inclusion.

### Q: Can I add my own models?
**A:** Yes! The script is designed to be extensible. See the Contributing section for guidelines on adding new models.

### Q: What about commercial models?
**A:** Commercial/proprietary models that require API access (GPT-4, Claude, etc.) aren't suitable for local deployment analysis. The script focuses on open-source models you can download and run yourself.

### Q: How accurate are the hardware requirements?
**A:** The requirements are based on real-world testing and community feedback. However, actual performance may vary based on your specific hardware configuration, other running applications, and usage patterns.

### Q: Can I generate reports programmatically?
**A:** Yes! The script supports command-line arguments for automated report generation, making it suitable for CI/CD pipelines or automated documentation workflows.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Support

If this script helped you find the right LLM for your system, please consider:

⭐ **Starring this repository** - it helps others discover this tool!

Found a bug or have a feature request? Please [open an issue](https://github.com/humaxai2025/LLMHardwareChecker/issues).

## 📞 Getting Help

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/humaxai2025/LLMHardwareChecker/issues)
- **Feature Requests**: [GitHub Issues](https://github.com/humaxai2025/LLMHardwareChecker/issues) with "enhancement" label
- **Questions**: Check the FAQ section above or open a discussion issue

## 🗺️ Roadmap

### **Version 2.0 (Planned)**
- **Performance benchmarking**: Real-time speed testing for selected models
- **Storage calculator**: Detailed disk space analysis and recommendations
- **Power consumption analysis**: Battery and electricity usage estimates
- **Cloud cost calculator**: Compare local vs cloud deployment costs
- **Model comparison matrix**: Side-by-side performance comparisons

### **Version 2.1 (Future)**
- **Interactive web interface**: Browser-based GUI for the tool
- **Model quality database**: Community ratings and performance scores
- **Multi-GPU support**: Analysis for systems with multiple GPUs
- **Container deployment**: Docker and Kubernetes configuration generation

---

**Made with ❤️ by HumanXAi for the AI community. Happy LLM running! 🤖**

---

### 📊 Script Statistics
- **Models Analyzed**: 17+ LLMs (12 general-purpose + 5 specialized)
- **Domain Coverage**: General purpose, Code generation, Medical/Biology, Mathematics, Research/Analysis
- **Report Formats**: HTML and PDF with professional styling
- **Platforms Supported**: 4 installation methods (Ollama, HuggingFace, GGUF, LM Studio)
- **Hardware Types**: CPU, GPU (NVIDIA/AMD/Apple), RAM, Storage
- **Operating Systems**: Windows, macOS, Linux
- **Specialization Focus**: Truly deployable domain-specific models only