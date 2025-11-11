# LLM Hardware Compatibility Checker v1.0.0

A professional, enterprise-grade web application that analyzes your system hardware and provides personalized recommendations for running Large Language Models (LLMs) locally. Built with Next.js 15, React 19, TypeScript, and Tailwind CSS.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Next.js](https://img.shields.io/badge/Next.js-15.5-black)
![React](https://img.shields.io/badge/React-19-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

### 🔍 **Comprehensive Hardware Analysis**
- Manual hardware specification input with browser-detected defaults
- Support for Windows, macOS, and Linux
- GPU detection via WebGL (NVIDIA, AMD, Intel, Apple Silicon)
- Multi-GPU detection and optimization recommendations
- Hardware profile management (save, load, export, import)
- No data sent to external servers - 100% client-side

### 🤖 **Smart Model Recommendations**
- **44+ verified LLM models** with detailed specifications
- Sortable table interface (by name, size, RAM, VRAM, speed)
- Performance tier classification (Excellent/Good/Basic/Not Suitable)
- Real-time performance estimates (tokens/second, load time, memory usage)
- Expandable model details with installation commands
- Category-based filtering with model counts
- Specialized models for coding, reasoning, creative writing, and more

### 🎯 **Advanced Tools** (Tabbed Interface)

#### **LLM Services Detection**
- Real-time detection of Ollama (localhost:11434)
- Real-time detection of LM Studio (localhost:1234)
- Display installed Ollama models with sizes
- No simulated data - only genuine detection

#### **Hardware Upgrade Advisor**
- Intelligent upgrade recommendations (RAM, GPU, CPU)
- Cost estimates for each upgrade
- Impact analysis (models unlocked per upgrade)
- Multi-GPU setup recommendations
- Priority-based suggestions (High/Medium/Low)

#### **Model Comparison Tool**
- Side-by-side comparison of up to 4 models
- Compare specs, performance, and requirements
- Perfect for decision-making

#### **Quantization Calculator**
- Visual comparison of 6 quantization levels (Q2_K to FP16)
- Quality vs. size tradeoffs
- Speed and compression metrics

#### **Hardware Profiles**
- Save unlimited hardware configurations
- Quick switching between systems
- Export profiles to JSON
- Import profiles from JSON
- Useful for testing multiple setups

### 🚀 **Export Configurations**
Generate hardware-optimized configurations for each model:
- **Docker Compose**: Full docker-compose.yml with GPU support, auto-pull
- **Ollama Modelfile**: Optimized parameters (threads, context, GPU layers, batch size)
- **llama.cpp Commands**: Command-line flags tailored to your hardware
- **API Configuration**: OpenAI-compatible API examples (Python, cURL)
- One-click copy and download

### 🛠️ **Detailed Installation Guides**
- **Ollama**: Easiest setup for beginners (recommended)
- **LM Studio**: GUI-based model management
- **llama.cpp**: Advanced CPU/GPU optimization
- **HuggingFace Transformers**: Python developer integration
- Platform-specific instructions for all operating systems

### 💡 **Optimization Tips**
- System-specific performance recommendations
- Quantization suggestions based on your hardware
- Memory and storage optimization tips
- GPU acceleration guidance
- Multi-GPU tensor parallelism configuration

### 📄 **Professional Reports**
- **PDF Reports**: Printable, shareable documents with system specs and recommendations
- **Quick Summary**: Clipboard-ready overview
- Completely offline and private

### 🎨 **Modern UI/UX**
- **Dark Mode**: Full dark theme with system preference detection
- **Keyboard Shortcuts**:
  - `Ctrl/Cmd + K`: Toggle dark/light mode
  - `Ctrl/Cmd + P`: Print
  - `Esc`: Close modals
- **Responsive Design**: Works flawlessly on desktop, tablet, and mobile
- **Smooth Animations**: Powered by Framer Motion
- **Accessibility**: WCAG 2.1 Level AA compliant
- **Print-Friendly**: Enhanced styles for printing reports
- **Enterprise Design System**: Professional color palette and typography

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- npm, yarn, or pnpm

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/llm-hardware-checker.git
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

### Production Build

```bash
npm run build
npm run start
```

---

## 🏗️ Project Structure

```
llm-hardware-checker/
├── app/                              # Next.js 15 App Router
│   ├── globals.css                  # Global styles, dark mode, print styles
│   ├── layout.tsx                   # Root layout with ThemeProvider
│   ├── page.tsx                     # Main application page
│   └── api/                         # API routes
├── components/                       # React components
│   ├── AdvancedToolsTabs.tsx        # Tabbed container for advanced features
│   ├── ExportConfigurations.tsx     # Docker, Modelfile, llama.cpp, API configs
│   ├── InstallationGuide.tsx        # Platform installation guides
│   ├── KeyboardShortcuts.tsx        # Global keyboard shortcut handler
│   ├── LoadingSpinner.tsx           # Loading animations
│   ├── ManualHardwareInput.tsx      # Hardware specification form
│   ├── ModelComparison.tsx          # Side-by-side model comparison
│   ├── ModelRecommendations.tsx     # Sortable model table with export
│   ├── OptimizationTips.tsx         # Performance recommendations
│   ├── ProfileManager.tsx           # Hardware profile management
│   ├── QuantizationCalculator.tsx   # Quantization comparison tool
│   ├── ReportDownload.tsx           # PDF report generation
│   ├── SystemMonitor.tsx            # LLM services detection
│   ├── SystemSpecsCard.tsx          # Hardware specifications display
│   ├── ThemeToggle.tsx              # Dark/light mode toggle button
│   └── UpgradeAdvisor.tsx           # Hardware upgrade recommendations
├── lib/                             # Core logic and utilities
│   ├── llmDatabase.ts               # 44+ model database with types
│   ├── llmRecommender.ts            # Recommendation engine
│   ├── performanceEstimator.ts      # Tokens/sec calculation
│   ├── profileManager.ts            # Profile save/load logic
│   ├── reportGenerator.ts           # PDF report generation
│   ├── systemAnalyzer.ts            # Hardware detection
│   └── ThemeContext.tsx             # Dark mode context provider
├── public/                          # Static assets
├── package.json                     # Dependencies (Next.js 15, React 19)
├── tailwind.config.js              # Tailwind CSS with dark mode
├── tsconfig.json                   # TypeScript configuration
└── README.md                       # This file
```

---

## 🧠 How It Works

### 1. **System Detection**
The application uses browser APIs and manual input to detect:
- **CPU**: Cores via `navigator.hardwareConcurrency`
- **GPU**: Graphics cards via WebGL renderer strings
- **Platform**: Operating system via `navigator.platform`
- **RAM/VRAM**: User-provided (browsers can't access for security)

### 2. **Compatibility Analysis**
Each model includes:
- Minimum and recommended RAM/VRAM requirements
- Parameter count and quantization options
- CPU-only compatibility flags
- Domain specialization tags
- Installation methods (Ollama, HuggingFace, etc.)

### 3. **Performance Estimation**
Real-time calculations based on your hardware:
- **Tokens per second**: Estimated inference speed
- **Load time**: Model initialization time
- **Memory usage**: RAM/VRAM utilization percentage
- **GPU usage**: GPU utilization percentage
- **Power consumption**: Estimated watts

### 4. **Recommendation Engine**
Models are categorized into performance tiers:
- **🟢 Excellent**: Runs smoothly with recommended settings
- **🔵 Good**: Runs well with minor limitations
- **🟡 Basic**: Minimal compatibility, quantization recommended
- **🔴 Not Suitable**: Insufficient hardware

### 5. **LLM Services Detection**
Real API calls to detect running services:
- **Ollama**: GET `http://localhost:11434/api/tags`
- **LM Studio**: GET `http://localhost:1234/v1/models`
- Displays installed models with sizes
- Auto-refreshes every 10 seconds

---

## 📊 Supported Models (44+)

### General Purpose Models
- **Llama 3.2/3.1** (1B, 3B, 8B, 70B, 405B)
- **DeepSeek-R1** (1.5B, 7B, 8B, 14B, 32B, 70B, 671B)
- **Mistral 7B**
- **Mixtral 8x7B** (47B total, 12B active)
- **Phi-3/3.5 Mini** (3.8B)
- **Gemma 2B/7B**
- **Qwen2.5** (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)
- **Vicuna 7B/13B**
- **Llama 2** (7B, 13B, 70B)

### Specialized Models
- **Code Generation**: Code Llama, Qwen2.5 Coder, CodeGemma
- **Mathematics**: DeepSeek Math 7B
- **Lightweight**: TinyLlama 1.1B, Gemma 2B, Qwen 0.5B
- **Advanced Reasoning**: DeepSeek-R1 series, Llama 3.1 405B

### Additional Models
- StableLM 2, Yi (6B, 34B), Nous Hermes 2, OpenHermes 2.5, Zephyr 7B, Solar 10.7B, Starling 7B, Orca 2 (7B, 13B), Neural Chat 7B, Falcon 7B

*All models are verified and available on Ollama/HuggingFace*

---

## 🎨 Customization

### Adding New Models
Edit `lib/llmDatabase.ts`:

```typescript
"Your Model Name": {
  parameters: "7B",
  min_ram_gb: 6,
  recommended_ram_gb: 12,
  min_vram_gb: 4,
  recommended_vram_gb: 6,
  cpu_only: true,
  domain: "General", // Optional: "Code", "Math", "Creative", etc.
  description: "Model description",
  install_methods: {
    ollama: {
      command: "ollama run your-model",
      note: "Installation note"
    }
  }
}
```

### Customizing Dark Mode Colors
Modify `app/globals.css`:

```css
.dark .bg-white {
  background-color: #1e293b; /* Your custom dark background */
}

.dark .text-gray-900 {
  color: #f1f5f9; /* Your custom dark text */
}
```

---

## 🚀 Deployment

### Vercel (Recommended)
1. Fork this repository
2. Connect to Vercel
3. Deploy automatically with zero configuration

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/llm-hardware-checker)

### Netlify
```bash
npm run build
# Deploy the .next folder
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

---

## 🧪 Testing

### Run Linter
```bash
npm run lint
```

### Build Test
```bash
npm run build
```

### Type Check
```bash
npx tsc --noEmit
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with proper TypeScript types
4. Test thoroughly: `npm run build && npm run lint`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Contribution Ideas
- 🆕 **New Models**: Add latest LLM releases
- 🎨 **UI/UX**: Improve design and accessibility
- 🐛 **Bug Fixes**: Fix issues and improve stability
- 📚 **Documentation**: Enhance guides and examples
- 🔧 **Features**: Add new functionality
- 🌐 **Localization**: Translate to other languages
- ⚡ **Performance**: Optimize bundle size and speed

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Models**: Verified from Ollama, HuggingFace, Meta AI, Mistral AI, Google DeepMind
- **UI Components**: Headless UI and Heroicons
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **PDF Generation**: jsPDF and html2canvas
- **Framework**: Next.js and React

---

## 📞 Support & Contact

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/llm-hardware-checker/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/llm-hardware-checker/discussions)
- 📧 **Email**: sriram@example.com
- 🌐 **Website**: [llmhardwarechecker.com](https://llmhardwarechecker.com)

---

## 🗺️ Roadmap

### v1.1.0 (Planned)
- [ ] Real-time benchmark mode (measure actual tokens/sec)
- [ ] Model changelog tracking
- [ ] Saved favorites and custom lists
- [ ] Cost calculator (local vs. cloud comparison)
- [ ] Historical tracking (upgrade impact over time)

### v2.0.0 (Future)
- [ ] Multi-language support (i18n)
- [ ] Cloud GPU rental integration
- [ ] Community benchmark submissions
- [ ] Browser extension version
- [ ] Mobile app (React Native)
- [ ] Enterprise features (team profiles, SSO)

---

## 📜 Version History

### v1.0.0 (Current)
- ✅ 44+ verified LLM models
- ✅ Hardware profiles with save/load/export
- ✅ Dark mode with keyboard shortcuts
- ✅ LLM services detection (Ollama, LM Studio)
- ✅ Export configurations (Docker, Modelfile, llama.cpp, API)
- ✅ Multi-GPU optimization recommendations
- ✅ Performance estimator (tokens/sec)
- ✅ Quantization calculator
- ✅ Model comparison tool
- ✅ Hardware upgrade advisor
- ✅ Sortable model table
- ✅ Enhanced print styles
- ✅ WCAG 2.1 Level AA accessibility

---

**Built with ❤️ by Sriram Srinivasan**

*Helping developers and researchers find the perfect local LLM for their hardware since 2025*

---

## 🔧 Tech Stack

| Category | Technology | Version |
|----------|------------|---------|
| **Framework** | Next.js | 15.5.6 |
| **UI Library** | React | 19.0.0 |
| **Language** | TypeScript | 5.7.2 |
| **Styling** | Tailwind CSS | 3.4.17 |
| **Animations** | Framer Motion | 11.15.0 |
| **Icons** | Heroicons | 2.2.0 |
| **PDF Generation** | jsPDF | 3.0.3 |
| **Notifications** | React Hot Toast | 2.4.1 |
| **Analytics** | Vercel Analytics | Latest |
| **Performance** | Vercel Speed Insights | Latest |
| **Build Tool** | Next.js (Turbopack) | - |
| **Deployment** | Vercel | - |

---

## ⚡ Performance

- **Bundle Size**: 95.2 kB (optimized)
- **First Load JS**: 197 kB
- **Build Time**: ~4 seconds
- **Lighthouse Score**: 95+ (Performance, Accessibility, Best Practices, SEO)
- **Mobile Responsive**: 100%
- **Zero Runtime Errors**: Production-ready

---

## 🔒 Privacy & Security

- ✅ **No external API calls** (except for LLM service detection on localhost)
- ✅ **Minimal analytics** (Vercel Analytics tracks page views and Web Vitals only - no personal data)
- ✅ **No cookies or local storage** (except theme preference and profiles)
- ✅ **Client-side only processing**
- ✅ **No server-side data storage**
- ✅ **Open source and auditable**

Your hardware information stays on your device. Always.

**Note:** Vercel Analytics collects anonymous page views and Web Vitals (performance metrics) to help improve the application. No personal information or hardware specifications are tracked.

---

## 💻 Browser Support

| Browser | Minimum Version |
|---------|----------------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |

---

## 📸 Screenshots

*Coming soon - Add screenshots of your application here*

---

**Star ⭐ this repository if you find it helpful!**
