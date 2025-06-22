import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  DocumentArrowDownIcon,
  DocumentTextIcon,
  ClipboardDocumentListIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { SystemInfo } from '../lib/systemAnalyzer';
import { Recommendations } from '../lib/llmDatabase';
import { LLMRecommender } from '../lib/llmRecommender';
import { ReportGenerator } from '../lib/reportGenerator';

interface ReportDownloadProps {
  systemInfo: SystemInfo;
  recommendations: Recommendations;
  recommender: LLMRecommender;
}

const ReportDownload: React.FC<ReportDownloadProps> = ({
  systemInfo,
  recommendations,
  recommender
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedReports, setGeneratedReports] = useState<{
    html?: string;
    pdf?: boolean;
  }>({});

  const generateHTMLReport = async () => {
    setIsGenerating(true);
    toast.loading('Generating HTML report...', { id: 'html-report' });

    try {
      const reportGenerator = new ReportGenerator({
        systemInfo,
        recommendations,
        recommender,
        timestamp: new Date()
      });

      const htmlContent = reportGenerator.generateHTMLReport();

      const blob = new Blob([htmlContent], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `llm-compatibility-report-${new Date().toISOString().split('T')[0]}.html`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setGeneratedReports(prev => ({ ...prev, html: htmlContent }));
      toast.success('HTML report downloaded!', { id: 'html-report' });
    } catch (error) {
      console.error('Failed to generate HTML report:', error);
      toast.error('Failed to generate HTML report', { id: 'html-report' });
    } finally {
      setIsGenerating(false);
    }
  };

  const generatePDFReport = async () => {
    setIsGenerating(true);
    toast.loading('Generating comprehensive PDF report...', { id: 'pdf-report' });

    try {
      const { jsPDF } = await import('jspdf');
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 15;
      const contentWidth = pageWidth - margin * 2;
      let y = margin;

      // Helper functions for consistent formatting
      const heading = (text: string, size = 16, color: [number, number, number] = [0, 0, 0]) => {
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(size);
        pdf.setTextColor(color[0], color[1], color[2]);
        // Remove emojis and clean text for PDF compatibility
        const cleanText = text.replace(/[^\x00-\x7F]/g, '').trim();
        pdf.text(cleanText, margin, y);
        y += size * 0.7 + 3;
      };

      const subheading = (text: string, size = 13, color: [number, number, number] = [0, 0, 0]) => {
        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(size);
        pdf.setTextColor(color[0], color[1], color[2]);
        const cleanText = text.replace(/[^\x00-\x7F]/g, '').trim();
        pdf.text(cleanText, margin, y);
        y += size * 0.7 + 2;
      };

      const para = (text: string, size = 10, indent = 0) => {
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(size);
        pdf.setTextColor(0, 0, 0);
        const cleanText = text.replace(/[^\x00-\x7F]/g, '').trim();
        const lines = pdf.splitTextToSize(cleanText, contentWidth - indent);
        pdf.text(lines, margin + indent, y);
        y += lines.length * size * 0.5 + 2;
      };

      const bulletPoint = (text: string, size = 10, indent = 5) => {
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(size);
        pdf.setTextColor(0, 0, 0);
        const cleanText = text.replace(/[^\x00-\x7F]/g, '').trim();
        const lines = pdf.splitTextToSize(`• ${cleanText}`, contentWidth - indent);
        pdf.text(lines, margin + indent, y);
        y += lines.length * size * 0.5 + 1;
      };

      const code = (text: string, size = 9) => {
        pdf.setFont('courier', 'normal');
        pdf.setFontSize(size);
        pdf.setFillColor(245, 245, 245);
        pdf.setTextColor(0, 0, 0);
        const cleanText = text.replace(/[^\x00-\x7F]/g, '').trim();
        const lines = pdf.splitTextToSize(cleanText, contentWidth - 10);
        const height = lines.length * size * 0.5 + 6;
        pdf.rect(margin, y, contentWidth, height, 'F');
        pdf.setDrawColor(200, 200, 200);
        pdf.rect(margin, y, contentWidth, height, 'S');
        pdf.text(lines, margin + 3, y + 4);
        y += height + 3;
      };

      const ensureSpace = (requiredHeight: number) => {
        if (y + requiredHeight > pageHeight - margin) {
          pdf.addPage();
          y = margin;
        }
      };

      const drawSeparator = () => {
        pdf.setDrawColor(180, 180, 180);
        pdf.line(margin, y + 2, pageWidth - margin, y + 2);
        y += 6;
      };

      // TITLE PAGE WITH HEADER
      pdf.setFillColor(37, 99, 235);
      pdf.rect(0, 0, pageWidth, 30, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(22);
      pdf.setFont('helvetica', 'bold');
      pdf.text('LLM Hardware Compatibility Report', margin, 18);
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(12);
      pdf.text(`Generated: ${new Date().toLocaleString()}`, margin, 25);
      y = 40;
      pdf.setTextColor(0, 0, 0);

      // EXECUTIVE SUMMARY
      heading('EXECUTIVE SUMMARY', 16, [37, 99, 235]);
      const suitableModels = [
        ...recommendations.excellent,
        ...recommendations.good,
        ...recommendations.basic
      ];
      bulletPoint(`Total Compatible Models: ${suitableModels.length}`);
      bulletPoint(`Excellent Performance: ${recommendations.excellent.length} models`);
      bulletPoint(`Good Performance: ${recommendations.good.length} models`);
      bulletPoint(`Basic Performance: ${recommendations.basic.length} models`);
      bulletPoint(`Not Suitable: ${recommendations.not_suitable.length} models`);
      
      if (suitableModels.length > 0) {
        y += 3;
        para(`RECOMMENDED: ${suitableModels[0].name} (${suitableModels[0].specs.parameters} parameters)`, 11);
        if (suitableModels[0].specs.install_methods.ollama) {
          para(`Quick Start Command: ${suitableModels[0].specs.install_methods.ollama.command}`, 10, 5);
        }
      } else {
        para('Your system does not meet minimum requirements for local LLMs. Consider cloud-based solutions.', 11);
      }
      y += 8;

      // SYSTEM SPECIFICATIONS
      ensureSpace(15);
      heading('SYSTEM SPECIFICATIONS', 16, [37, 99, 235]);
      
      // Create a clean specification layout
      pdf.setFillColor(248, 250, 252);
      pdf.rect(margin, y, contentWidth, 35, 'F');
      pdf.setDrawColor(226, 232, 240);
      pdf.rect(margin, y, contentWidth, 35, 'S');
      
      y += 5;
      subheading('Hardware Configuration', 12);
      bulletPoint(`Operating System: ${systemInfo.os} (${systemInfo.architecture})`);
      bulletPoint(`Processor: ${systemInfo.processor} with ${systemInfo.cpuCores} cores`);
      bulletPoint(`Total RAM: ${systemInfo.totalRamGB} GB (${systemInfo.availableRamGB} GB available)`);
      bulletPoint(`Storage: ${systemInfo.freeStorageGB} GB free of ${systemInfo.totalStorageGB} GB total`);
      
      if (systemInfo.gpus && systemInfo.gpus.length > 0) {
        systemInfo.gpus.forEach((gpu, i) => {
          const vramInfo = typeof gpu.vramGB === 'number' ? `${gpu.vramGB} GB VRAM` : gpu.vramGB;
          bulletPoint(`Graphics Card ${i + 1}: ${gpu.name} (${vramInfo})`);
        });
      } else {
        bulletPoint('Graphics: No dedicated GPU detected - CPU-only processing');
      }
      y += 8;

      // COMPATIBILITY SUMMARY
      ensureSpace(15);
      heading('COMPATIBILITY ANALYSIS', 16, [37, 99, 235]);
      
      if (suitableModels.length > 0) {
        para('ANALYSIS RESULT: Your system can run local LLMs effectively!', 12);
        y += 2;
        
        // Performance capability breakdown
        const capability = recommender.getSystemCapabilityLevel();
        const capabilityMap = {
          low: 'Entry Level - Basic models recommended',
          medium: 'Mid Range - Good selection of 7B models available', 
          high: 'High End - Can run larger 13B+ models efficiently',
          premium: 'Premium - Capable of running largest available models'
        };
        
        para(`System Performance Level: ${capabilityMap[capability] || 'Unknown'}`, 11);
        y += 3;
        
        // Quick recommendations
        subheading('Recommended Starting Points:', 12);
        bulletPoint('Beginners: Start with Gemma 2B or Llama 3.2 3B for fast performance');
        bulletPoint('Coding: Try Code Llama 7B or StarCoder 7B for programming tasks');
        bulletPoint('Advanced: Llama 3.1 8B offers excellent general capabilities');
        bulletPoint('Specialized: BioMistral (medical) or MetaMath (mathematics) for domain-specific needs');
      } else {
        para('ANALYSIS RESULT: Your system does not meet minimum requirements for local LLMs.', 12);
        y += 2;
        para('Recommendation: Use cloud-based AI services like ChatGPT, Claude, or Gemini.', 11);
      }
      y += 8;

      // DETAILED MODEL RECOMMENDATIONS
      const categories = [
        { 
          title: 'EXCELLENT PERFORMANCE MODELS', 
          models: recommendations.excellent, 
          color: [34, 197, 94] as [number, number, number],
          description: 'These models will run smoothly with great performance on your system'
        },
        { 
          title: 'GOOD PERFORMANCE MODELS', 
          models: recommendations.good, 
          color: [234, 179, 8] as [number, number, number],
          description: 'These models will work well with acceptable performance'
        },
        { 
          title: 'BASIC PERFORMANCE MODELS', 
          models: recommendations.basic, 
          color: [239, 68, 68] as [number, number, number],
          description: 'These models will work but may have slower performance'
        }
      ];

      for (const category of categories) {
        if (category.models.length > 0) {
          ensureSpace(20);
          
          // Category header with colored background
          pdf.setFillColor(category.color[0], category.color[1], category.color[2]);
          pdf.rect(margin, y, contentWidth, 8, 'F');
          pdf.setTextColor(255, 255, 255);
          pdf.setFont('helvetica', 'bold');
          pdf.setFontSize(14);
          pdf.text(category.title, margin + 2, y + 5);
          y += 10;
          
          pdf.setTextColor(0, 0, 0);
          para(category.description, 10);
          y += 3;
          
          category.models.forEach((model, idx) => {
            ensureSpace(25);
            
            // Model header box
            pdf.setFillColor(249, 250, 251);
            pdf.setDrawColor(209, 213, 219);
            pdf.rect(margin, y, contentWidth, 6, 'FD');
            
            pdf.setFont('helvetica', 'bold');
            pdf.setFontSize(12);
            pdf.setTextColor(31, 41, 55);
            pdf.text(`${idx + 1}. ${model.name}`, margin + 2, y + 4);
            y += 8;
            
            // Model details
            pdf.setFont('helvetica', 'normal');
            pdf.setFontSize(10);
            pdf.setTextColor(0, 0, 0);
            
            if (model.specs.domain) {
              bulletPoint(`Specialized Domain: ${model.specs.domain}`, 10, 3);
            }
            bulletPoint(`Model Size: ${model.specs.parameters} parameters`, 10, 3);
            bulletPoint(`Performance: ${model.compatibility.performance_tier}`, 10, 3);
            bulletPoint(`Description: ${model.specs.description}`, 10, 3);
            
            y += 2;
            subheading('System Requirements:', 11);
            bulletPoint(`RAM: ${model.specs.min_ram_gb} GB minimum, ${model.specs.recommended_ram_gb} GB recommended`, 9, 6);
            bulletPoint(`VRAM: ${model.specs.min_vram_gb || 'N/A'} GB minimum, ${model.specs.recommended_vram_gb || 'N/A'} GB recommended`, 9, 6);
            bulletPoint(`CPU-Only Support: ${model.specs.cpu_only ? 'Yes' : 'No'}`, 9, 6);
            
            if (model.compatibility.recommended_quant) {
              bulletPoint(`Recommended Quantization: ${model.compatibility.recommended_quant}`, 9, 6);
            }
            
            if (model.compatibility.notes.length > 0) {
              bulletPoint(`Special Notes: ${model.compatibility.notes.join('; ')}`, 9, 6);
            }
            
            y += 2;
            subheading('Installation Options:', 11);
            
            // Installation methods
            if (model.specs.install_methods.ollama) {
              para('Option 1: OLLAMA (Recommended for Beginners)', 10, 3);
              code(model.specs.install_methods.ollama.command);
              para(`Setup: ${model.specs.install_methods.ollama.note}`, 9, 6);
              y += 1;
            }
            
            if (model.specs.install_methods.lm_studio) {
              para('Option 2: LM STUDIO (GUI Application)', 10, 3);
              if (model.specs.install_methods.lm_studio.command) {
                code(model.specs.install_methods.lm_studio.command);
              }
              para(`Setup: ${model.specs.install_methods.lm_studio.note}`, 9, 6);
              y += 1;
            }
            
            if (model.specs.install_methods.huggingface) {
              para('Option 3: HUGGING FACE (For Developers)', 10, 3);
              if (model.specs.install_methods.huggingface.model_id) {
                para(`Model ID: ${model.specs.install_methods.huggingface.model_id}`, 9, 6);
              }
              if (model.specs.install_methods.huggingface.command) {
                code(model.specs.install_methods.huggingface.command);
              }
              para(`Setup: ${model.specs.install_methods.huggingface.note}`, 9, 6);
              y += 1;
            }
            
            if (model.specs.install_methods.gguf) {
              para('Option 4: GGUF/llama.cpp (Advanced Users)', 10, 3);
              if (model.specs.install_methods.gguf.source) {
                para(`Download: ${model.specs.install_methods.gguf.source}`, 8, 6);
              }
              if (model.specs.install_methods.gguf.recommended_quant) {
                para(`Recommended: ${model.specs.install_methods.gguf.recommended_quant}`, 9, 6);
              }
              para(`Notes: ${model.specs.install_methods.gguf.note}`, 9, 6);
              y += 1;
            }
            
            if (model.specs.install_methods.llamacpp) {
              para('Option 5: LLAMA.CPP (Command Line)', 10, 3);
              if (model.specs.install_methods.llamacpp.command) {
                code(model.specs.install_methods.llamacpp.command);
              }
              if (model.specs.install_methods.llamacpp.download_url) {
                para(`Download: ${model.specs.install_methods.llamacpp.download_url}`, 8, 6);
              }
              para(`Notes: ${model.specs.install_methods.llamacpp.note}`, 9, 6);
            }
            
            drawSeparator();
          });
        }
      }

      // NOT SUITABLE MODELS
      if (recommendations.not_suitable.length > 0) {
        ensureSpace(12);
        heading('MODELS NOT SUITABLE FOR YOUR HARDWARE', 16, [220, 38, 127]);
        para('These models require more powerful hardware than your system provides:', 11);
        y += 2;
        
        recommendations.not_suitable.forEach((model, idx) => {
          const ramReq = `${model.specs.recommended_ram_gb}GB RAM`;
          const vramReq = model.specs.recommended_vram_gb ? `${model.specs.recommended_vram_gb}GB VRAM` : 'N/A VRAM';
          bulletPoint(`${model.name} - Requires ${ramReq}, ${vramReq}`, 10, 3);
        });
        y += 5;
      }

      // PLATFORM INSTALLATION GUIDE
      ensureSpace(20);
      heading('PLATFORM INSTALLATION GUIDE', 16, [37, 99, 235]);
      
      subheading('1. Ollama (Recommended for Beginners)', 13, [59, 130, 246]);
      para('Ollama is the easiest way to run LLMs locally with automatic model management.', 11);
      bulletPoint('Website: https://ollama.ai', 10, 5);
      bulletPoint('Windows: Download installer from website', 10, 5);
      bulletPoint('macOS: Download installer or use: brew install ollama', 10, 5);
      bulletPoint('Linux: curl -fsSL https://ollama.ai/install.sh | sh', 10, 5);
      bulletPoint('Usage: ollama run <model-name>', 10, 5);
      y += 3;

      subheading('2. LM Studio (GUI Application)', 13, [59, 130, 246]);
      para('User-friendly graphical interface for running LLMs without command line.', 11);
      bulletPoint('Website: https://lmstudio.ai', 10, 5);
      bulletPoint('Available for Windows, macOS, and Linux', 10, 5);
      bulletPoint('Features: Model browser, chat interface, API server', 10, 5);
      y += 3;

      subheading('3. llama.cpp (Advanced Users)', 13, [59, 130, 246]);
      para('High-performance C++ implementation for maximum efficiency.', 11);
      bulletPoint('Repository: https://github.com/ggerganov/llama.cpp', 10, 5);
      bulletPoint('Requires compilation from source or pre-built binaries', 10, 5);
      bulletPoint('Best performance for CPU-only inference', 10, 5);
      bulletPoint('Supports GGUF quantized models', 10, 5);
      y += 3;

      subheading('4. HuggingFace Transformers (Developers)', 13, [59, 130, 246]);
      para('Python library for machine learning developers and researchers.', 11);
      bulletPoint('Installation: pip install transformers torch', 10, 5);
      bulletPoint('Requires Python programming knowledge', 10, 5);
      bulletPoint('Access to largest model repository', 10, 5);
      bulletPoint('Full customization and fine-tuning capabilities', 10, 5);
      y += 8;

      // OPTIMIZATION TIPS
      ensureSpace(20);
      heading('SYSTEM OPTIMIZATION TIPS', 16, [37, 99, 235]);
      para('Personalized recommendations based on your hardware configuration:', 11);
      y += 3;
      
      const tips = recommender.getOptimizationTips();
      tips.forEach((tip, idx) => {
        const cleanTip = tip.replace(/^[^\w\s]+\s*/, ''); // Remove emoji prefixes
        bulletPoint(`${cleanTip}`, 10, 3);
      });
      y += 8;

      // QUICK START GUIDE
      if (suitableModels.length > 0) {
        ensureSpace(15);
        heading('QUICK START GUIDE', 16, [37, 99, 235]);
        const topModel = suitableModels[0];
        para('Follow these steps to get your first local LLM running:', 12);
        y += 3;
        
        para('Step 1: Download and Install Ollama', 11, 3);
        bulletPoint('Visit https://ollama.ai and download for your operating system', 10, 8);
        bulletPoint('Run the installer and follow setup instructions', 10, 8);
        y += 2;
        
        para('Step 2: Install Your First Model', 11, 3);
        if (topModel.specs.install_methods.ollama) {
          code(topModel.specs.install_methods.ollama.command);
        }
        bulletPoint('Wait for the model to download (first time only)', 10, 8);
        bulletPoint('The model will be cached locally for future use', 10, 8);
        y += 2;
        
        para('Step 3: Start Using Your LLM', 11, 3);
        bulletPoint('Once downloaded, you can immediately start chatting', 10, 8);
        bulletPoint('Type your questions and press Enter', 10, 8);
        bulletPoint('Type /bye to exit the chat session', 10, 8);
        y += 3;
        
        subheading('Useful Commands:', 12);
        code('ollama list              # See installed models');
        code('ollama rm <model>        # Remove a model to free space');
        code('ollama help              # View all available commands');
        y += 5;
      } else {
        ensureSpace(10);
        heading('CLOUD-BASED ALTERNATIVES', 16, [37, 99, 235]);
        para('Since your hardware cannot run local LLMs effectively, consider these cloud alternatives:', 11);
        y += 3;
        bulletPoint('ChatGPT (OpenAI): https://chat.openai.com - Most popular AI assistant', 10, 5);
        bulletPoint('Claude (Anthropic): https://claude.ai - Helpful and safe AI assistant', 10, 5);
        bulletPoint('Gemini (Google): https://gemini.google.com - Google\'s AI with web search', 10, 5);
        bulletPoint('Perplexity AI: https://perplexity.ai - AI-powered search and research', 10, 5);
        y += 5;
      }

      // TROUBLESHOOTING
      ensureSpace(15);
      heading('TROUBLESHOOTING COMMON ISSUES', 16, [37, 99, 235]);
      
      subheading('Out of Memory Errors:', 12, [185, 28, 28]);
      bulletPoint('Try smaller quantized models (Q4_K_M instead of Q8_0)', 10, 5);
      bulletPoint('Close other applications to free up RAM', 10, 5);
      bulletPoint('Use CPU-only mode if GPU memory is insufficient', 10, 5);
      y += 3;

      subheading('Slow Performance:', 12, [185, 28, 28]);
      bulletPoint('Enable GPU acceleration if available', 10, 5);
      bulletPoint('Use appropriately sized models for your hardware', 10, 5);
      bulletPoint('Ensure sufficient free storage space', 10, 5);
      bulletPoint('Close unnecessary background applications', 10, 5);
      y += 3;

      subheading('Installation Issues:', 12, [185, 28, 28]);
      bulletPoint('Verify your system meets minimum requirements', 10, 5);
      bulletPoint('Check internet connection for model downloads', 10, 5);
      bulletPoint('Consult platform-specific documentation', 10, 5);
      bulletPoint('Try running as administrator if permissions are denied', 10, 5);
      y += 8;

      // COMPATIBILITY MATRIX TABLE
      pdf.addPage();
      y = margin;
      heading('DETAILED COMPATIBILITY MATRIX', 16, [37, 99, 235]);
      para('Complete overview of all models and their compatibility with your system:', 11);
      y += 5;
      
      // Create comprehensive table
      const allModels = [
        ...recommendations.excellent,
        ...recommendations.good,
        ...recommendations.basic,
        ...recommendations.not_suitable,
      ];

      // Table header with background
      pdf.setFillColor(59, 130, 246);
      pdf.rect(margin, y, contentWidth, 8, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(10);
      
      // Column widths and headers
      const colWidths = [45, 18, 25, 18, 18, 20, 20, 26];
      const headers = ['Model Name', 'Size', 'Performance', 'Min RAM', 'Rec RAM', 'Min VRAM', 'Rec VRAM', 'Compatibility'];
      let x = margin;
      
      headers.forEach((header, i) => {
        pdf.text(header, x + 1, y + 5);
        x += colWidths[i];
      });
      y += 10;

      // Table rows
      pdf.setTextColor(0, 0, 0);
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(8);
      
      allModels.forEach((model, index) => {
        ensureSpace(5);
        
        // Alternate row colors
        if (index % 2 === 0) {
          pdf.setFillColor(248, 250, 252);
          pdf.rect(margin, y, contentWidth, 6, 'F');
        }
        
        x = margin;
        const isCompatible = [...recommendations.excellent, ...recommendations.good, ...recommendations.basic].includes(model);
        const status = isCompatible ? 'Compatible' : 'Not Suitable';
        
        const rowData = [
          model.name.length > 25 ? model.name.substring(0, 22) + '...' : model.name,
          model.specs.parameters,
          model.compatibility.performance_tier.length > 15 ? 
            model.compatibility.performance_tier.substring(0, 12) + '...' : model.compatibility.performance_tier,
          `${model.specs.min_ram_gb}GB`,
          `${model.specs.recommended_ram_gb}GB`,
          `${model.specs.min_vram_gb || 'N/A'}GB`,
          `${model.specs.recommended_vram_gb || 'N/A'}GB`,
          status
        ];
        
        rowData.forEach((cell, i) => {
          pdf.text(String(cell), x + 1, y + 4, { maxWidth: colWidths[i] - 2 });
          x += colWidths[i];
        });
        y += 6;
      });

      // Summary statistics
      y += 8;
      heading('COMPATIBILITY SUMMARY', 14, [37, 99, 235]);
      
      pdf.setFillColor(240, 253, 244);
      pdf.rect(margin, y, contentWidth, 25, 'F');
      pdf.setDrawColor(34, 197, 94);
      pdf.rect(margin, y, contentWidth, 25, 'S');
      
      y += 4;
      subheading('Model Compatibility Statistics', 12);
      bulletPoint(`Total Models Analyzed: ${allModels.length}`, 11, 5);
      bulletPoint(`Compatible Models: ${suitableModels.length} (${Math.round((suitableModels.length / allModels.length) * 100)}%)`, 11, 5);
      bulletPoint(`Excellent Performance: ${recommendations.excellent.length} models`, 11, 5);
      bulletPoint(`Good Performance: ${recommendations.good.length} models`, 11, 5);
      bulletPoint(`Basic Performance: ${recommendations.basic.length} models`, 11, 5);
      bulletPoint(`Not Suitable: ${recommendations.not_suitable.length} models`, 11, 5);
      y += 8;

      // Hardware utilization analysis
      if (suitableModels.length > 0) {
        subheading('Hardware Utilization Analysis', 12);
        const ramUtilization = Math.round((suitableModels[0].specs.recommended_ram_gb / systemInfo.totalRamGB) * 100);
        const hasGPU = systemInfo.gpus && systemInfo.gpus.length > 0;
        
        bulletPoint(`RAM Utilization for recommended model: ~${ramUtilization}%`, 10, 5);
        if (hasGPU) {
          bulletPoint('GPU acceleration available for better performance', 10, 5);
          bulletPoint('Can run higher quality quantizations (Q8_0 or FP16)', 10, 5);
        } else {
          bulletPoint('CPU-only processing - consider GPU upgrade for better performance', 10, 5);
        }
        bulletPoint(`Storage space needed: 2-15GB per model depending on size`, 10, 5);
      }
      
      y += 10;

      // FINAL RECOMMENDATIONS
      heading('FINAL RECOMMENDATIONS', 16, [37, 99, 235]);
      
      if (suitableModels.length > 0) {
        subheading('Next Steps for Success:', 13, [34, 197, 94]);
        bulletPoint('Start with our top recommendation to ensure smooth experience', 11, 5);
        bulletPoint('Install Ollama first - it\'s the most beginner-friendly option', 11, 5);
        bulletPoint('Test with smaller models before trying larger ones', 11, 5);
        bulletPoint('Monitor system resources during usage', 11, 5);
        bulletPoint('Join LLM communities for tips and troubleshooting help', 11, 5);
        y += 5;
        
        subheading('Performance Optimization:', 13, [234, 179, 8]);
        bulletPoint('Close unnecessary applications before running large models', 11, 5);
        bulletPoint('Use quantized models (Q4_K_M, Q5_K_M) for better performance', 11, 5);
        if (systemInfo.gpus && systemInfo.gpus.length > 0) {
          bulletPoint('GPU acceleration will significantly improve speed', 11, 5);
        }
        bulletPoint('Consider upgrading RAM if you want to run larger models', 11, 5);
      } else {
        subheading('Alternative Solutions:', 13, [239, 68, 68]);
        bulletPoint('Cloud-based AI services are your best option currently', 11, 5);
        bulletPoint('Consider hardware upgrades: more RAM and/or dedicated GPU', 11, 5);
        bulletPoint('Monitor developments in efficient model architectures', 11, 5);
        bulletPoint('Smaller edge models may become available in the future', 11, 5);
      }

      // FOOTER
      y = pageHeight - 25;
      pdf.setFillColor(248, 250, 252);
      pdf.rect(0, y, pageWidth, 25, 'F');
      pdf.setDrawColor(229, 231, 235);
      pdf.line(0, y, pageWidth, y);
      
      y += 8;
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(12);
      pdf.setTextColor(59, 130, 246);
      pdf.text('LLM Hardware Compatibility Checker', margin, y);
      
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(10);
      pdf.setTextColor(107, 114, 128);
      pdf.text('Complete analysis of your system capabilities for local LLM deployment', margin, y + 5);
      pdf.text(`Comprehensive report generated on ${new Date().toLocaleString()}`, margin, y + 10);
      
      // Right-aligned footer info
      pdf.text('Visit our website for updates and new model compatibility', pageWidth - margin - 80, y + 5);
      pdf.text('Report any issues or suggestions via the feedback system', pageWidth - margin - 80, y + 10);

      // Save the PDF
      pdf.save(`llm-compatibility-report-comprehensive-${new Date().toISOString().split('T')[0]}.pdf`);
      setGeneratedReports(prev => ({ ...prev, pdf: true }));
      toast.success('Comprehensive PDF report downloaded!', { id: 'pdf-report' });
    } catch (error) {
      console.error('Failed to generate comprehensive PDF report:', error);
      toast.error('Failed to generate PDF report. Try HTML format instead.', { id: 'pdf-report' });
    } finally {
      setIsGenerating(false);
    }
  };

  const copyReportSummary = async () => {
    const suitableModels = [
      ...recommendations.excellent,
      ...recommendations.good,
      ...recommendations.basic
    ];

    const summary = `
LLM Hardware Compatibility Report
Generated: ${new Date().toLocaleDateString()}

SYSTEM SPECIFICATIONS:
- OS: ${systemInfo.os} (${systemInfo.architecture})
- CPU: ${systemInfo.processor} (${systemInfo.cpuCores} cores)
- RAM: ${systemInfo.totalRamGB} GB total, ${systemInfo.availableRamGB} GB available
- Storage: ${systemInfo.freeStorageGB} GB free / ${systemInfo.totalStorageGB} GB total
- GPUs: ${systemInfo.gpus?.length || 0} detected
${systemInfo.gpus?.map(gpu => `  - ${gpu.name} (${typeof gpu.vramGB === 'number' ? gpu.vramGB + ' GB VRAM' : gpu.vramGB})`).join('\n') || ''}

COMPATIBILITY RESULTS:
- Compatible Models: ${suitableModels.length}
- Excellent Performance: ${recommendations.excellent.length}
- Good Performance: ${recommendations.good.length}  
- Basic Performance: ${recommendations.basic.length}
- Not Suitable: ${recommendations.not_suitable.length}

RECOMMENDED MODELS:
${suitableModels.slice(0, 5).map(model => 
  `- ${model.name}: ${model.specs.parameters} parameters (${model.compatibility.performance_tier})`
).join('\n')}

NEXT STEPS:
1. Install Ollama from https://ollama.ai
2. Run: ollama run ${suitableModels[0]?.name.toLowerCase().replace(/[^a-z0-9]/g, '') || 'llama3.2:3b'}
3. Start chatting with your local LLM!

For detailed installation instructions and optimization tips, download the full report.
`.trim();

    try {
      await navigator.clipboard.writeText(summary);
      toast.success('Report summary copied to clipboard!');
    } catch (error) {
      toast.error('Failed to copy to clipboard');
    }
  };

  const suitableModelsCount = [
    ...recommendations.excellent,
    ...recommendations.good,
    ...recommendations.basic
  ].length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden"
    >
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 px-6 py-4">
        <div className="flex items-center">
          <DocumentArrowDownIcon className="h-8 w-8 text-white mr-3" />
          <h2 className="text-2xl font-bold text-white">Download Detailed Report</h2>
        </div>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Report Options */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              📄 Report Formats Available
            </h3>
            
            <div className="space-y-4">
              {/* HTML Report */}
              <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <DocumentTextIcon className="h-6 w-6 text-blue-600 mr-3" />
                    <div>
                      <h4 className="font-semibold text-gray-900">HTML Report</h4>
                      <p className="text-sm text-gray-600">Interactive web report with all details</p>
                    </div>
                  </div>
                  {generatedReports.html && (
                    <CheckCircleIcon className="h-5 w-5 text-green-500" />
                  )}
                </div>
                
                <div className="flex flex-wrap gap-2 mb-3">
                  <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium">
                    Interactive
                  </span>
                  <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium">
                    Copy Commands
                  </span>
                  <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs font-medium">
                    Searchable
                  </span>
                </div>
                
                <button
                  onClick={generateHTMLReport}
                  disabled={isGenerating}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isGenerating ? 'Generating...' : 'Download HTML Report'}
                </button>
              </div>

              {/* PDF Report */}
              <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <DocumentArrowDownIcon className="h-6 w-6 text-red-600 mr-3" />
                    <div>
                      <h4 className="font-semibold text-gray-900">PDF Report</h4>
                      <p className="text-sm text-gray-600">Printable document format</p>
                    </div>
                  </div>
                  {generatedReports.pdf && (
                    <CheckCircleIcon className="h-5 w-5 text-green-500" />
                  )}
                </div>
                
                <div className="flex flex-wrap gap-2 mb-3">
                  <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs font-medium">
                    Printable
                  </span>
                  <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-xs font-medium">
                    Shareable
                  </span>
                  <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs font-medium">
                    Offline
                  </span>
                </div>
                
                <button
                  onClick={generatePDFReport}
                  disabled={isGenerating}
                  className="w-full bg-red-600 text-white py-2 px-4 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isGenerating ? 'Generating...' : 'Download PDF Report'}
                </button>
              </div>

              {/* Quick Summary */}
              <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center mb-3">
                  <ClipboardDocumentListIcon className="h-6 w-6 text-green-600 mr-3" />
                  <div>
                    <h4 className="font-semibold text-gray-900">Quick Summary</h4>
                    <p className="text-sm text-gray-600">Copy key findings to clipboard</p>
                  </div>
                </div>
                
                <button
                  onClick={copyReportSummary}
                  className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors"
                >
                  Copy Summary to Clipboard
                </button>
              </div>
            </div>
          </div>

          {/* Report Preview */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              📋 Report Contents Preview
            </h3>
            
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-4">
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">🖥️ System Specifications</h4>
                <p className="text-sm text-gray-600">Complete hardware analysis and detection results</p>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">🤖 Model Recommendations</h4>
                <p className="text-sm text-gray-600">
                  {suitableModelsCount} compatible models with performance tiers and installation instructions
                </p>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">🛠️ Installation Guide</h4>
                <p className="text-sm text-gray-600">Step-by-step setup for Ollama, LM Studio, llama.cpp, and HuggingFace</p>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">💡 Optimization Tips</h4>
                <p className="text-sm text-gray-600">System-specific recommendations for best performance</p>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-800">📊 Compatibility Matrix</h4>
                <p className="text-sm text-gray-600">Detailed breakdown of which models work with your hardware</p>
              </div>
            </div>
            
            {/* File Info */}
            <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-800 mb-2">📁 File Information</h4>
              <div className="text-sm text-blue-700 space-y-1">
                <p>• HTML: ~200-500 KB, works in any web browser</p>
                <p>• PDF: ~1-3 MB, compatible with all PDF viewers</p>
                <p>• Reports include all analysis data and instructions</p>
                <p>• Generated files are completely offline and private</p>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Options */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            🔗 Share & Save Options
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-800 mb-2">💾 Save for Later</h4>
              <p className="text-sm text-yellow-700 mb-3">
                Download reports to reference installation steps offline
              </p>
            </div>
            
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-semibold text-green-800 mb-2">📤 Share Results</h4>
              <p className="text-sm text-green-700 mb-3">
                Share compatibility results with team members or forums
              </p>
            </div>
            
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h4 className="font-semibold text-purple-800 mb-2">🔄 Compare Systems</h4>
              <p className="text-sm text-purple-700 mb-3">
                Run analysis on multiple systems and compare capabilities
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ReportDownload;