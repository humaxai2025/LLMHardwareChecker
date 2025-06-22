# LLM Hardware Compatibility Checker

A modern, elegant web application that analyzes your system hardware and provides personalized recommendations for running Large Language Models (LLMs) locally. Built with Next.js, TypeScript, and Tailwind CSS, deployed on Vercel.

![LLM Hardware Checker](https://img.shields.io/badge/LLM-Hardware%20Checker-blue)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3-blue)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-black)

## ✨ Features

### 🔍 **Comprehensive System Analysis**
- Real-time hardware detection (CPU, RAM, GPU, Storage)
- Browser-based analysis (no data sent to servers)
- Support for Windows, macOS, Linux, and mobile devices
- GPU detection including NVIDIA, AMD, Intel, and Apple Silicon

### 🤖 **Smart Model Recommendations**
- 15+ popular LLM models with detailed specifications
- Performance tier classification (Excellent/Good/Basic)
- Specialized models for coding, mathematics, medical, and research domains
- Compatibility checking based on your exact hardware

### 🛠️ **Detailed Installation Guides**
- **Ollama**: Easiest setup for beginners
- **LM Studio**: GUI-based model management
- **llama.cpp**: Advanced CPU/GPU optimization
- **HuggingFace Transformers**: Python developer integration
- Platform-specific instructions for all operating systems

### 💡 **Optimization Tips**
- System-specific performance recommendations
- Quantization suggestions (Q3_K_M, Q4_K_M, Q8_0, FP16)
- Memory and storage optimization tips
- GPU acceleration guidance

### 📄 **Professional Reports**
- **HTML Reports**: Interactive, searchable, with copy-paste commands
- **PDF Reports**: Printable, shareable documents
- **Quick Summary**: Clipboard-ready overview
- Completely offline and private

### 🎨 **Modern UI/UX**
- Responsive design that works on all devices
- Smooth animations and transitions
- Accessibility-first approach
- Dark mode support (coming soon)
- Global design standards compliance

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn or pnpm

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/llm-hardware-checker.git
cd llm-hardware-checker
```

2. **Install dependencies**
```bash
npm install
# or
yarn install
# or
pnpm install
```

3. **Run the development server**
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

4. **Open your browser**
Navigate to [http://localhost:3000](http://localhost:3000)

## 🏗️ Project Structure

```
llm-hardware-checker/
├── app/                          # Next.js 14 App Router
│   ├── globals.css              # Global styles and Tailwind
│   ├── layout.tsx               # Root layout component
│   └── page.tsx                 # Main application page
├── components/                   # React components
│   ├── InstallationGuide.tsx    # Platform installation guides
│   ├── LoadingSpinner.tsx       # Loading animations
│   ├── ModelRecommendations.tsx # Model cards and compatibility
│   ├── OptimizationTips.tsx     # Performance recommendations
│   ├── ReportDownload.tsx       # Report generation
│   └── SystemSpecsCard.tsx      # Hardware specifications display
├── lib/                         # Core logic and utilities
│   ├── llmDatabase.ts           # Model database and types
│   ├── llmRecommender.ts        # Recommendation engine
│   ├── reportGenerator.ts       # Report generation logic
│   └── systemAnalyzer.ts        # Hardware detection
├── public/                      # Static assets
├── package.json                 # Dependencies and scripts
├── tailwind.config.js          # Tailwind CSS configuration
├── tsconfig.json               # TypeScript configuration
└── README.md                   # This file
```

## 🧠 How It Works

### 1. **System Detection**
The application uses browser APIs to detect:
- **CPU**: Cores and architecture via `navigator.hardwareConcurrency`
- **Memory**: Available RAM via `navigator.deviceMemory` (where supported)
- **GPU**: Graphics cards via WebGL renderer strings
- **Storage**: Available space via Storage API
- **Platform**: Operating system and device type

### 2. **Compatibility Analysis**
Each model in the database includes:
- Minimum and recommended RAM/VRAM requirements
- Parameter count and model size
- CPU-only compatibility flags
- Specialized domain tags

### 3. **Recommendation Engine**
The system categorizes models into performance tiers:
- **🟢 Excellent**: Can run with recommended settings
- **🟡 Good**: Can run with some limitations
- **🟠 Basic**: Minimal compatibility, quantization required
- **🔴 Not Suitable**: Insufficient hardware

### 4. **Report Generation**
- **HTML**: Complete interactive report with embedded CSS/JS
- **PDF**: Generated using html2canvas and jsPDF
- **Summary**: Formatted text for quick sharing

## 📊 Supported Models

### General Purpose Models
- **Llama 3.1/3.2** (3B, 8B, 13B, 70B)
- **Mistral 7B** 
- **Phi-3 Mini** (3.8B)
- **Gemma 2B**
- **Vicuna 13B**

### Specialized Domain Models
- **Code Generation**: Code Llama, StarCoder, WizardCoder
- **Mathematics**: MetaMath 7B
- **Medical/Biology**: BioMistral 7B
- **Research/Analysis**: Nous Hermes 2 Solar

*Total: 15+ models with more being added regularly*

## 🔧 Customization

### Adding New Models
Edit `lib/llmDatabase.ts` to add new models:

```typescript
"Your Model Name": {
  parameters: "7B",
  min_ram_gb: 6,
  recommended_ram_gb: 12,
  min_vram_gb: 4,
  recommended_vram_gb: 6,
  cpu_only: true,
  domain: "Your Domain", // Optional
  description: "Model description",
  install_methods: {
    ollama: {
      command: "ollama run your-model",
      note: "Installation note"
    }
    // ... other methods
  }
}
```

### Customizing Themes
Modify `tailwind.config.js` for colors and styling:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        // Your custom color palette
      }
    }
  }
}
```

## 🚀 Deployment

### Vercel (Recommended)
1. Fork this repository
2. Connect to Vercel
3. Deploy automatically

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-username/llm-hardware-checker)

### Other Platforms
- **Netlify**: `npm run build && npm run export`
- **GitHub Pages**: Enable static export in `next.config.js`
- **Docker**: Dockerfile included for containerized deployment

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Test thoroughly: `npm run test` and `npm run lint`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Areas for Contribution
- 🆕 **New Models**: Add support for latest LLMs
- 🎨 **UI/UX**: Improve design and user experience  
- 🐛 **Bug Fixes**: Fix issues and improve stability
- 📚 **Documentation**: Improve guides and documentation
- 🔧 **Features**: Add new functionality
- 🌐 **Localization**: Translate to other languages

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Model Information**: Curated from official model repositories
- **Hardware Detection**: Uses modern browser APIs
- **UI Components**: Built with Headless UI and Heroicons
- **Styling**: Tailwind CSS for responsive design
- **Animations**: Framer Motion for smooth interactions

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-username/llm-hardware-checker/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/your-username/llm-hardware-checker/discussions)
- 📧 **Email**: support@llmhardwarechecker.com
- 💬 **Discord**: [Community Server](https://discord.gg/your-server)

## 🗺️ Roadmap

### Version 2.0 (Coming Soon)
- [ ] **Model Performance Benchmarks**: Real performance data
- [ ] **Cloud GPU Rental Integration**: Direct links to GPU cloud services
- [ ] **Model Size Calculator**: Precise storage requirements
- [ ] **Multi-language Support**: International accessibility
- [ ] **Advanced Filtering**: Filter by use case, domain, performance
- [ ] **System Monitoring**: Real-time resource usage tracking
- [ ] **Model Download Manager**: Direct model downloads with progress
- [ ] **Community Reviews**: User ratings and reviews for models

### Future Enhancements
- Dark mode theme
- PWA (Progressive Web App) support
- Browser extension version
- Mobile app (React Native)
- Enterprise features for organizations

---

**Made with ❤️ by HumanXAI for the open-source LLM community**

*Helping developers and researchers find the perfect local LLM for their hardware*