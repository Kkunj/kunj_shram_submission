import time
import psutil
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for each model"""
    name: str
    model_id: str
    size_gb: float
    description: str
    min_ram_gb: float
    recommended_ram_gb: float

@dataclass
class QuantizationOption:
    """Quantization configuration options"""
    name: str
    config: Optional[BitsAndBytesConfig]
    memory_multiplier: float  # How much memory this saves (1.0 = no savings)
    description: str


class InteractiveLLMBenchmark:
    def __init__(self):
        """Initialize the interactive benchmarking tool"""
        
        # Define available models with detailed specs
        self.available_models = [
            ModelConfig(
                name="Gemma-2B",
                model_id="google/gemma-2b",
                size_gb=4.0,
                description="Lightweight model, good for basic tasks",
                min_ram_gb=6.0,
                recommended_ram_gb=8.0
            ),
            ModelConfig(
                name="Qwen2.5-7B", 
                model_id="Qwen/Qwen2.5-7B",
                size_gb=14.0,
                description="Mid-size model, balanced performance/efficiency",
                min_ram_gb=16.0,
                recommended_ram_gb=24.0
            ),
            ModelConfig(
                name="Llama-3.1-8B",
                model_id="meta-llama/Llama-3.1-8B",
                size_gb=16.0,
                description="Large model, best performance",
                min_ram_gb=20.0,
                recommended_ram_gb=32.0
            )
        ]
        
        # Define quantization options (most efficient first)
        self.quantization_options = [
            QuantizationOption(
                name="4-bit",
                config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
                memory_multiplier=0.25,
                description="Most memory efficient, slight quality loss"
            ),
            QuantizationOption(
                name="8-bit", 
                config=BitsAndBytesConfig(load_in_8bit=True),
                memory_multiplier=0.5,
                description="Good balance of efficiency and quality"
            ),
            QuantizationOption(
                name="16-bit",
                config=None,
                memory_multiplier=1.0,
                description="Full precision, best quality, most memory"
            )
        ]
        
        # Test prompts of different lengths for comprehensive benchmarking
        self.test_prompts = {
            "short": "Explain AI in one sentence.",
            "medium": "Write a detailed explanation of artificial intelligence and its applications in modern technology.",
            "long": "Write a comprehensive essay about the impact of artificial intelligence on society, covering both benefits and challenges, including specific examples from healthcare, education, and business sectors."
        }
        
        self.results = []
        self.system_info = None
    
    def get_system_info(self) -> Dict:
        """Collect and display detailed system information"""
        print("🔍 Analyzing your system configuration...")
        print("-" * 50)
        
        # Get basic system info
        memory = psutil.virtual_memory()
        info = {
            "total_ram_gb": memory.total / (1024**3),
            "available_ram_gb": memory.available / (1024**3),
            "used_ram_gb": memory.used / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
            "has_gpu": torch.cuda.is_available(),
        }
        
        # GPU information
        if info["has_gpu"]:
            gpu_props = torch.cuda.get_device_properties(0)
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": gpu_props.total_memory / (1024**3),
                "gpu_compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
            })
        
        # Display system information
        print(f"💻 System Configuration:")
        print(f"   CPU: {info['cpu_count']} cores @ {info['cpu_freq']:.0f}MHz")
        print(f"   RAM: {info['total_ram_gb']:.1f}GB total | {info['available_ram_gb']:.1f}GB available | {info['used_ram_gb']:.1f}GB used")
        
        if info["has_gpu"]:
            print(f"   GPU: {info['gpu_name']}")
            print(f"        {info['gpu_memory_gb']:.1f}GB VRAM | Compute {info['gpu_compute_capability']}")
        else:
            print("   GPU: ❌ Not detected (CPU-only inference)")
            
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        free_space_gb = disk_usage.free / (1024**3)
        print(f"   Disk: {free_space_gb:.1f}GB free space")
        
        if free_space_gb < 40:
            print("   ⚠️  Warning: Low disk space! Models require ~34GB total")
        
        self.system_info = info
        return info

    def cleanup_model(self, model, tokenizer):
        """Clean up memory after each model"""
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  🧹 Memory cleaned up")
    
    def analyze_and_recommend(self):
        """Analyze results and provide recommendations"""
        if not self.results:
            print("❌ No successful benchmarks to analyze")
            return
        
        successful_results = [r for r in self.results if r.get("status") == "success"]
        
        if not successful_results:
            print("❌ No successful benchmarks completed")
            return
        
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS & ANALYSIS")
        print("=" * 60)
        
        # Display detailed results
        for result in successful_results:
            print(f"\n🎯 {result['model_name']} ({result['quantization']} quantization)")
            print(f"   Load Time: {result['load_time']:.1f}s")
            print(f"   Memory Usage: {result['ram_used_gb']:.1f}GB RAM + {result['gpu_memory_gb']:.1f}GB GPU")
            print("   Performance:")
            
            for prompt_type in ['short', 'medium', 'long']:
                if f"{prompt_type}_tpm" in result:
                    tpm = result[f"{prompt_type}_tpm"]
                    first_token = result[f"{prompt_type}_first_token_latency"] * 1000
                    total_latency = result[f"{prompt_type}_total_latency"]
                    print(f"     {prompt_type.capitalize():<8}: {tpm:6.1f} TPM | {first_token:4.0f}ms first token | {total_latency:.2f}s total")
        
        # Find best model
        best_model = max(successful_results, key=lambda x: x.get('medium_tpm', 0))
        
        print(f"\n🏆 RECOMMENDATION")
        print("-" * 30)
        print(f"Best Overall: {best_model['model_name']} ({best_model['quantization']})")
        print(f"Reasons:")
        print(f"  • Highest medium prompt performance: {best_model['medium_tpm']:.1f} TPM")
        print(f"  • Memory efficient: {best_model['ram_used_gb']:.1f}GB RAM usage")
        print(f"  • Fast loading: {best_model['load_time']:.1f}s startup time")
        
        # Save detailed results
        self.save_detailed_results()
    
    def download_and_load_model(self, model: ModelConfig) -> Tuple[Optional[object], Optional[object], Dict]:
        """Download and load model with comprehensive error handling for all systems"""
        print(f"\n📥 Processing {model.name}...")
        
        # Find best quantization option for this system
        compatibility = self.check_basic_compatibility(model)
        if not compatibility["options"]:
            return None, None, {"error": "Model incompatible with system"}
        
        # Detect system capabilities
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()  # Apple Silicon
        
        print(f"  🖥️  System: {'CUDA GPU' if has_cuda else 'MPS (Apple)' if has_mps else 'CPU-only'}")
        
        # Try quantization options in order of efficiency
        for quant_option in self.quantization_options:
            # Check if this quantization is compatible
            compatible_option = next(
                (opt for opt in compatibility["options"] if opt["quantization"] == quant_option.name), 
                None
            )
            
            if not compatible_option:
                continue
            
            # Skip quantization on CPU-only systems if it's problematic
            if not has_cuda and not has_mps and quant_option.name in ["4-bit", "8-bit"]:
                try:
                    # Test if bitsandbytes works on this system
                    import bitsandbytes as bnb
                    # Some systems don't support bitsandbytes without CUDA
                except ImportError:
                    print(f"  ⚠️  Skipping {quant_option.name} - bitsandbytes not available on CPU-only system")
                    continue
                
            try:
                print(f"  🔄 Attempting {quant_option.name} quantization...")
                print(f"     Expected RAM usage: {compatible_option['required_ram']:.1f}GB")
                
                # Show live system stats during download (non-blocking)
                try:
                    monitor_thread = threading.Thread(target=self._monitor_system_during_load, daemon=True)
                    monitor_thread.start()
                except Exception:
                    pass  # Continue without monitoring if threading fails
                
                # Load tokenizer first (smaller download)
                print("     📚 Loading tokenizer...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model.model_id, 
                        trust_remote_code=True,
                        use_auth_token=True  # Use saved auth token
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception as e:
                    print(f"     ❌ Tokenizer failed: {str(e)[:100]}...")
                    continue
                
                # Determine device strategy
                device_map = self._get_device_map(has_cuda, has_mps, quant_option)
                torch_dtype = self._get_torch_dtype(has_cuda, has_mps, quant_option)
                
                # Load main model
                print("     🧠 Loading model (this may take several minutes)...")
                start_time = time.time()
                
                try:
                    model_obj = AutoModelForCausalLM.from_pretrained(
                        model.model_id,
                        quantization_config=quant_option.config,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                        use_auth_token=True,  # Use saved auth token
                        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                        offload_folder="./offload" if not has_cuda and not has_mps else None,  # CPU offloading
                    )
                except Exception as model_error:
                    print(f"     ❌ Model loading failed: {str(model_error)[:150]}...")
                    
                    # Try fallback: CPU-only without quantization
                    if quant_option.config is not None:
                        print(f"     🔄 Trying CPU fallback without quantization...")
                        try:
                            model_obj = AutoModelForCausalLM.from_pretrained(
                                model.model_id,
                                device_map="cpu",
                                torch_dtype=torch.float32,  # Use float32 for CPU
                                trust_remote_code=True,
                                use_auth_token=True,
                                low_cpu_mem_usage=True,
                            )
                            quant_option.name = "CPU-float32"  # Update for reporting
                        except Exception as fallback_error:
                            print(f"     ❌ CPU fallback also failed: {str(fallback_error)[:100]}...")
                            continue
                    else:
                        continue
                
                load_time = time.time() - start_time
                
                # Verify model is actually loaded and functional
                try:
                    # Quick test to ensure model works
                    test_input = tokenizer("Test", return_tensors="pt")
                    if str(model_obj.device) != "cpu":
                        test_input = {k: v.to(model_obj.device) for k, v in test_input.items()}
                    
                    with torch.no_grad():
                        _ = model_obj(**test_input)
                    
                    print(f"     ✅ Model loaded successfully in {load_time:.1f}s with {quant_option.name} quantization")
                    print(f"     📍 Device: {model_obj.device}")
                    
                    return model_obj, tokenizer, {
                        "quantization": quant_option.name,
                        "load_time": load_time,
                        "device": str(model_obj.device),
                        "error": None
                    }
                    
                except Exception as test_error:
                    print(f"     ❌ Model loaded but failed functionality test: {str(test_error)[:100]}...")
                    # Clean up failed model
                    del model_obj
                    gc.collect()
                    if has_cuda:
                        torch.cuda.empty_cache()
                    continue
                    
            except Exception as e:
                print(f"     ❌ Failed with {quant_option.name}: {str(e)[:100]}...")
                continue
        
        return None, None, {"error": "Failed to load with any quantization option"}
        
