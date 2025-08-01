# Interactive LLM Local Benchmarking Tool

**NAME: KUNJ KANZARIYA**  
**EMAIL: kanzariyakunj1909@gmail.com**  
**Chosen Option: Option 1 (Local LLM setup)**  
Kaggle link: https://www.kaggle.com/code/kunjkanzariya/shram-benchmark

**Hoping for the best results 🤞**

## Description

This tool provides comprehensive benchmarking for Local Large Language Models (LLMs) with automatic hardware detection and optimization. It supports multiple quantization strategies, cross-platform compatibility (CPU-only, CUDA, Apple Silicon MPS), and provides detailed performance analysis with hardware-specific recommendations.

The benchmark tests the mentioned models (Gemma-2B, Qwen2.5-7B, Llama-3.1-8B) across different prompt lengths and quantization configurations, delivering actionable insights for optimal LLM deployment on your hardware.

## Code Flow Overview

### 1. **System Analysis**
- Detects hardware configuration (CPU, RAM, GPU type)
- Checks CUDA/MPS availability and quantization library support
- Analyzes system compatibility with available models

### 2. **Model Selection & Configuration**
- Displays compatible models based on hardware capabilities
- Shows quantization options (4-bit, 8-bit, GPU-float16, CPU-float32)
- Provides hardware-specific recommendations

### 3. **Model Downloading and Loading**
- Implements fallback strategies for different hardware configurations
- Handles quantization setup with error recovery
- Monitors system resources during loading process

### 4. **Performance Benchmarking**
- Tests models with three prompt types (short, medium, long)
- Measures tokens per minute (TPM), tokens per second (TPS), and latency
- Tracks memory usage (RAM and GPU VRAM)

### 5. **Analysis & Recommendations**
- Compares performance across different configurations
- Provides hardware-optimized recommendations
- Exports detailed results to CSV files for further analysis

## Prerequisites

1. **Python Environment**: Python 3.8+
2. **Dependencies**: Install from `requirements.txt`
   ```bash
   pip install -r requirements.txt
   ```
3. **Hugging Face Token**: Set up authentication for model access
   ```bash
   huggingface-cli login
   ```

## Usage

```bash
python interactive_llm_benchmark.py
```

Follow the interactive prompts to:
1. Review your system configuration
2. Select models to benchmark
3. Choose quantization strategies
4. Monitor real-time benchmarking progress
5. Analyze results and recommendations

## Major Problems Faced and Solutions

### 1. **Cross-Platform Quantization Compatibility**

**Problem**: Systems without GPU support (especially CPU-only machines) cannot run the `bitsandbytes` library, preventing 4-bit/8-bit quantization and causing application crashes.

**Solution**: 
- Implemented comprehensive system configuration detection
- Added multiple fallback strategies with graceful degradation
- Created universal CPU-compatible float32 quantization as ultimate fallback
- Added extensive error handling and recovery mechanisms throughout the codebase
- Integrated real-time hardware compatibility checks before model loading

This ensures the tool runs smoothly on all hardware configurations, from high-end multi-GPU setups to basic CPU-only systems, while automatically selecting the optimal configuration for each platform.

## Output

The tool generates:
- **Real-time console output** with system monitoring and progress updates
- **Detailed CSV reports** with performance metrics and hardware information
- **Hardware-specific recommendations** for optimal model deployment
- **Comparative analysis** across different model-quantization combinations

## Supported Hardware

- ✅ **CPU-only systems** (with automatic fallback optimization)
- ✅ **NVIDIA CUDA GPUs** (with 4-bit/8-bit quantization)
- ✅ **Apple Silicon (M1/M2/M3)** (with MPS optimization)
- ✅ **Multi-GPU setups** (with automatic device mapping)

---

**🤞 EXCITED TO INTERACT WITH YOU IN THE INTERVIEW 🤞**
