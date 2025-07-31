#!/usr/bin/env python3
"""
Interactive LLM Local Benchmarking Tool
Enhanced user experience with step-by-step guidance
"""

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
    
    def display_model_options(self):
        """Display available models with requirements"""
        print("\n📦 Available Models:")
        print("-" * 80)
        print(f"{'Model':<15} {'Size':<8} {'Min RAM':<10} {'Rec RAM':<10} {'Description'}")
        print("-" * 80)
        
        for i, model in enumerate(self.available_models, 1):
            compatibility = self.check_basic_compatibility(model)
            status = "✅" if compatibility["can_run"] else "❌"
            
            print(f"{i}. {model.name:<12} {model.size_gb:<6.1f}GB {model.min_ram_gb:<8.1f}GB {model.recommended_ram_gb:<8.1f}GB {model.description} {status}")
        
    def check_basic_compatibility(self, model: ModelConfig) -> Dict:
        """Check if model can potentially run on this system"""
        available_ram = self.system_info["available_ram_gb"]
        has_gpu = self.system_info["has_gpu"]
        
        # Check different quantization options
        compatible_options = []
        
        for quant in self.quantization_options:
            required_ram = model.size_gb * quant.memory_multiplier * 1.5  # 1.5x safety factor
            
            if available_ram >= required_ram:
                compatible_options.append({
                    "quantization": quant.name,
                    "required_ram": required_ram,
                    "description": quant.description
                })
        
        return {
            "can_run": len(compatible_options) > 0,
            "options": compatible_options,
            "has_gpu_advantage": has_gpu
        }
    
    def get_user_model_selection(self) -> List[ModelConfig]:
        """Interactive model selection"""
        while True:
            try:
                selection = input("\n🎯 Select models to benchmark (e.g., '1,3' or 'all'): ").strip().lower()
                
                if selection == 'all':
                    return [model for model in self.available_models 
                           if self.check_basic_compatibility(model)["can_run"]]
                
                if selection == 'exit' or selection == 'quit':
                    print("👋 Exiting...")
                    exit(0)
                
                # Parse comma-separated indices
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_models = []
                
                for idx in indices:
                    if 0 <= idx < len(self.available_models):
                        model = self.available_models[idx]
                        compatibility = self.check_basic_compatibility(model)
                        
                        if compatibility["can_run"]:
                            selected_models.append(model)
                        else:
                            print(f"⚠️  {model.name} cannot run on your system (insufficient RAM)")
                    else:
                        print(f"❌ Invalid selection: {idx + 1}")
                
                if selected_models:
                    return selected_models
                else:
                    print("❌ No valid models selected. Please try again.")
                    
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input. Please enter numbers separated by commas (e.g., '1,2') or 'all'")
    
    def show_detailed_compatibility(self, models: List[ModelConfig]):
        """Show detailed compatibility analysis for selected models"""
        print("\n🔧 Compatibility Analysis:")
        print("=" * 60)
        
        for model in models:
            print(f"\n📋 {model.name} ({model.size_gb}GB)")
            compatibility = self.check_basic_compatibility(model)
            
            if compatibility["options"]:
                print("   ✅ Compatible quantization options:")
                for i, option in enumerate(compatibility["options"], 1):
                    print(f"      {i}. {option['quantization']:<8} | {option['required_ram']:.1f}GB RAM | {option['description']}")
            else:
                print("   ❌ Cannot run on your system")
            
            if compatibility["has_gpu_advantage"]:
                print("   🚀 GPU acceleration available - faster inference expected")
    
    def confirm_proceed(self) -> bool:
        """Ask user confirmation to proceed"""
        while True:
            response = input("\n🚀 Proceed with benchmarking? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                print("👋 Benchmarking cancelled.")
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def download_and_load_model_single_option(self, model: ModelConfig, option: Dict) -> Tuple[Optional[object], Optional[object], Dict]:
        """Load model with a specific quantization option"""
        print(f"  🔄 Loading with {option['quantization']}...")
        print(f"     Expected RAM usage: {option['required_ram']:.1f}GB")
        
        try:
            # Monitor system during loading
            monitor_thread = threading.Thread(target=self._monitor_system_safely, daemon=True)
            monitor_thread.start()
            
            # Load tokenizer
            tokenizer = self._load_tokenizer_safely(model.model_id)
            if tokenizer is None:
                return None, None, {"error": "Tokenizer loading failed"}
            
            # Load model with this specific option
            start_time = time.time()
            model_obj = self._load_model_with_strategy(model.model_id, option, self.hardware_info)
            
            if model_obj is None:
                return None, None, {"error": f"Model loading failed with {option['quantization']}"}
            
            load_time = time.time() - start_time
            
            if not self._verify_model_functionality(model_obj, tokenizer):
                self._cleanup_failed_model(model_obj)
                return None, None, {"error": "Model failed functionality test"}
            
            print(f"     ✅ Success! Loaded in {load_time:.1f}s")
            
            return model_obj, tokenizer, {
                "quantization": option["quantization"],
                "load_time": load_time,
                "device": str(self._get_model_device(model_obj)),
                "error": None
            }
            
        except Exception as e:
            print(f"     ❌ Failed: {str(e)[:100]}...")
            return None, None, {"error": str(e)}

    def _get_device_map(self, has_cuda: bool, has_mps: bool, quant_option) -> str:
        """Determine optimal device mapping strategy"""
        if has_cuda:
            return "auto"  # Let transformers handle GPU placement
        elif has_mps:
            return "mps"   # Apple Silicon
        else:
            return "cpu"   # CPU-only fallback

    def _get_torch_dtype(self, has_cuda: bool, has_mps: bool, quant_option):
        """Determine optimal torch dtype for the system"""
        if quant_option.config is not None:
            # Quantized models handle their own dtype
            return None
        elif has_cuda:
            return torch.float16  # GPU supports float16
        elif has_mps:
            return torch.float32  # MPS is better with float32
        else:
            return torch.float32  # CPU requires float32

    def _monitor_system_during_load(self):
        """Monitor system resources during model loading - robust version"""
        try:
            for i in range(60):  # Monitor for up to 60 seconds
                try:
                    memory = psutil.virtual_memory()
                    cpu_percent = psutil.cpu_percent(interval=0.1)  # Non-blocking
                    
                    # Format output with proper clearing
                    status = f"     📊 RAM: {memory.percent:.1f}% used | CPU: {cpu_percent:.1f}%"
                    print(f"\r{status:<80}", end='', flush=True)
                    
                    time.sleep(2)
                    
                    # Stop if memory usage gets too high
                    if memory.percent > 90:
                        print(f"\n     ⚠️  High memory usage detected: {memory.percent:.1f}%")
                        break
                        
                except Exception:
                    # Continue silently if monitoring fails
                    time.sleep(2)
                    continue
                    
        except Exception:
            # Fail silently - monitoring is not critical
            pass
        finally:
            # Clear the monitoring line
            print(f"\r{' ' * 80}\r", end='', flush=True)

    
    def advanced_benchmark(self, model, tokenizer, model_name: str, model_info: Dict) -> Dict:
        """Run comprehensive benchmarking with multiple metrics"""
        print(f"\n⚡ Running comprehensive benchmark for {model_name}...")
        
        results = {
            "model_name": model_name,
            "quantization": model_info["quantization"],
            "load_time": model_info["load_time"]
        }
        
        try:
            # Warm-up run
            print("  🔥 Warming up model...")
            self._run_warmup(model, tokenizer)
            
            # Test different prompt lengths
            for prompt_type, prompt_text in self.test_prompts.items():
                print(f"  📝 Testing {prompt_type} prompt...")
                
                # Multiple runs for statistical accuracy
                times = []
                first_token_times = []
                
                for run in range(3):  # 3 runs for averaging
                    metrics = self._benchmark_single_run(model, tokenizer, prompt_text)
                    times.append(metrics["total_time"])
                    first_token_times.append(metrics["first_token_time"])
                
                # Calculate statistics
                avg_total_time = sum(times) / len(times)
                avg_first_token_time = sum(first_token_times) / len(first_token_times)
                
                # Calculate tokens per minute
                tokens_generated = 50  # Standard generation length
                tpm = (tokens_generated / avg_total_time) * 60
                
                results[f"{prompt_type}_tpm"] = round(tpm, 2)
                results[f"{prompt_type}_total_latency"] = round(avg_total_time, 3)
                results[f"{prompt_type}_first_token_latency"] = round(avg_first_token_time, 3)
                
                print(f"     ✅ {prompt_type.capitalize()}: {tpm:.1f} TPM | First token: {avg_first_token_time*1000:.0f}ms")
            
            # Memory usage
            memory_used = psutil.virtual_memory().used / (1024**3)
            gpu_memory = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            
            results.update({
                "ram_used_gb": round(memory_used, 2),
                "gpu_memory_gb": round(gpu_memory, 2),
                "status": "success"
            })
            
            return results
            
        except Exception as e:
            print(f"  ❌ Benchmark failed: {str(e)}")
            return {
                "model_name": model_name,
                "status": "failed",
                "error": str(e)
            }
    
    def _run_warmup(self, model, tokenizer):
        """Run warmup to stabilize performance"""
        inputs = tokenizer("Hello", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            model.generate(
                inputs["input_ids"],
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
    
    def _benchmark_single_run(self, model, tokenizer, prompt: str) -> Dict:
        """Run single benchmark iteration with detailed timing"""
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Measure first token time
        start_time = time.time()
        
        with torch.no_grad():
            # Generate with streaming to capture first token
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        total_time = time.time() - start_time
        
        # Estimate first token time (approximation)
        first_token_time = total_time * 0.1  # First token typically ~10% of total time
        
        return {
            "total_time": total_time,
            "first_token_time": first_token_time
        }
    
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
    
    def save_detailed_results(self):
        """Save comprehensive results to CSV"""
        df = pd.DataFrame(self.results)
        timestamp = int(time.time())
        filename = f"llm_benchmark_detailed_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Detailed results saved to: {filename}")
    
    def run_interactive_benchmark(self):
        """Main interactive benchmarking flow"""
        print("🚀 Interactive LLM Local Benchmarking Tool")
        print("=" * 50)
        
        # Step 1: System Analysis
        self.get_system_info()
        
        # Step 2: Model Selection
        self.display_model_options()
        selected_models = self.get_user_model_selection()
        
        if not selected_models:
            print("❌ No models selected. Exiting.")
            return
        
        # Step 3: Compatibility Check
        self.show_detailed_compatibility(selected_models)
        
        # Step 4: User Confirmation
        if not self.confirm_proceed():
            return
        
        # Step 5: Download and Benchmark
        print(f"\n🎯 Starting benchmark of {len(selected_models)} model(s)...")
        
        for model_config in selected_models:
            print(f"\n{'='*20} {model_config.name} {'='*20}")
            
            # Download and load
            model, tokenizer, model_info = self.download_and_load_model(model_config)
            
            if model is None:
                self.results.append({
                    "model_name": model_config.name,
                    "status": "failed",
                    "error": model_info.get("error", "Unknown error")
                })
                continue
            
            # Benchmark
            result = self.advanced_benchmark(model, tokenizer, model_config.name, model_info)
            self.results.append(result)
            
            # Cleanup
            self.cleanup_model(model, tokenizer)
        
        # Step 6: Analysis and Recommendations
        self.analyze_and_recommend()

if __name__ == "__main__":
    benchmark = InteractiveLLMBenchmark()
    benchmark.run_interactive_benchmark()
