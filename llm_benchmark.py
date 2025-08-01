#!/usr/bin/env python3
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
# import warnings


#Please change this field and add your token to run the code!!
#o get a Hugging Face API token, you need to sign in to your Hugging Face account (or create one), go to your settings, navigate to the "Access Tokens" tab, and then generate a new token, providing it with a name and a role (read or write). 

from huggingface_hub import login
#############################################
login(token="hf_----------------------NVKJyJ")
#############################################

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
    """Quantization configuration options with hardware compatibility"""
    name: str
    config: Optional[BitsAndBytesConfig]
    memory_multiplier: float
    description: str
    requires_cuda: bool = False  
    requires_gpu: bool = False   
    fallback_dtype: Optional[torch.dtype] = None  

@dataclass
class HardwareInfo:
    """Comprehensive hardware information"""
    has_cuda: bool
    has_mps: bool  # for Apple devices
    cuda_device_count: int
    gpu_names: List[str]
    gpu_memory_gb: List[float]
    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    supports_bitsandbytes: bool

class InteractiveLLMBenchmark:
    def __init__(self):
        """Initialize the benchmarking tool"""
        
        # required specs are taken from the huggingface.
        # Source: https://huggingface.co/google/gemma-2b/discussions/61
        self.available_models = [
            ModelConfig(
                name="Gemma-2B",
                model_id="google/gemma-2b",
                size_gb=4.0,
                description="Lightweight model, good for basic tasks",
                min_ram_gb=6.0,
                recommended_ram_gb=8.0
            ),
            # Source: https://blogs.novita.ai/qwen-2-5-7b-vram/
            ModelConfig(
                name="Qwen2.5-7B", 
                model_id="Qwen/Qwen2.5-7B",
                size_gb=14.0,
                description="Mid-size model, balanced performance/efficiency",
                min_ram_gb=16.0,
                recommended_ram_gb=24.0
            ),
            #Source: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/77
            ModelConfig(
                name="Llama-3.1-8B",
                model_id="meta-llama/Llama-3.1-8B",
                size_gb=16.0,
                description="Large model, best performance",
                min_ram_gb=20.0,
                recommended_ram_gb=32.0
            )
        ]
        
        #have to create another fucntion to handle systems without GPU support.
        self.quantization_options = self.quantization_options()
        
        
        self.test_prompts = {
            "short": "Explain how one can win in a technical assessment round in one sentence.",
            
            "medium": "Describe, in 2–3 sentences, how your technical interview performance was so smooth it felt like you were debugging their questions in real time. Include a subtle hint that anyone doubting your skills might need a firmware upgrade themselves.",
            
            "long": "Write a humorous 7–8 sentence account of a technical interview where your answers were so precise, your logic so flawless, and your confidence so contagious that the panel started questioning their own understanding of the topic. Mention how you predicted edge cases before they even asked, optimized code they hadn’t shown yet, and even spotted a typo in their own test cases. Add a touch of drama—perhaps a stunned silence after one of your answers. Throw in a joke about whether they were evaluating you or silently taking notes for their next promotion. Let the narrative subtly suggest that rejecting you would be less of a hiring decision and more of a historic blunder. Finish with a playful nod to your humility—something like, 'But hey, I’m just here to learn.'"
        }
        
        self.results = []
        self.hardware_info = None

    def quantization_options(self) -> List[QuantizationOption]:
        """Initialize quantization options with hardware compatibility checks"""
        
        #Added this function to print and convey to user the status of bitsandbytes.
        bitsandbytes_available = self._check_bitsandbytes_support()
        
        options = []
        
        # 4-bit quantization (most memory efficient, but comes with performance degration)
        if bitsandbytes_available:
            try:
                config_4bit = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",  
                    bnb_4bit_use_double_quant=True  
                )
                options.append(QuantizationOption(
                    name="4-bit",
                    config=config_4bit,
                    memory_multiplier=0.25,
                    description="Most memory efficient, requires CUDA",
                    requires_cuda=True,
                    requires_gpu=True
                ))
            except Exception as e:
                print(f"OOOOPS, GOT ERROR: 4-bit quantization not available: {e}")
        
        # 8-bit quantization (balanced)
        if bitsandbytes_available:
            try:
                config_8bit = BitsAndBytesConfig(load_in_8bit=True)
                options.append(QuantizationOption(
                    name="8-bit",
                    config=config_8bit,
                    memory_multiplier=0.5,
                    description="Good balance, requires CUDA",
                    requires_cuda=True,
                    requires_gpu=True
                ))
            except Exception as e:
                print(f"OOOOPS, GOT ERROR: 8-bit quantization not available: {e}")
        
        # GPU float16 (no quantization but GPU optimized)
        options.append(QuantizationOption(
            name="GPU-float16",
            config=None,
            memory_multiplier=0.6,  # float16 saves some memory vs float32
            description="GPU-optimized float16, requires GPU",
            requires_cuda=False,
            requires_gpu=True,
            fallback_dtype=torch.float16
        ))
        
        # CPU float32 (universal fallback)
        options.append(QuantizationOption(
            name="CPU-float32",
            config=None,
            memory_multiplier=1.0,
            description="CPU-compatible, works on all systems",
            requires_cuda=False,
            requires_gpu=False,
            fallback_dtype=torch.float32
        ))
         
        return options

    def _check_bitsandbytes_support(self) -> bool:
        """Check if bitsandbytes library is available and functional"""
        try:
            import bitsandbytes as bnb
        
            if not torch.cuda.is_available():
                print("INFO:  bitsandbytes available but CUDA not detected - quantization disabled")
                return False

            test_config = BitsAndBytesConfig(load_in_8bit=True)
            print("INFO: bitsandbytes fully functional")
            return True
            
        except ImportError:
            print("OOOOPS, GOT ERROR:  bitsandbytes not installed - quantization features disabled")
            return False
        except Exception as e:
            print(f"OOOOPS, GOT ERROR:  bitsandbytes available but not functional: {e}")
            return False

    def get_hardware_info(self) -> HardwareInfo:
        """Collect hardware information"""

        print("Analyzing your system configuration...")
        print("-" * 50)
        
        # Basic system info
        memory = psutil.virtual_memory()
        
        # GPU Detection
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        cuda_device_count = 0
        gpu_names = []
        gpu_memory_gb = []
        
        if has_cuda:
            cuda_device_count = torch.cuda.device_count()
            for i in range(cuda_device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_mem = gpu_props.total_memory / (1024**3)
                gpu_names.append(gpu_name)
                gpu_memory_gb.append(gpu_mem)
        elif has_mps:
            
            gpu_names.append("Apple Silicon GPU")
            gpu_memory_gb.append(0)  
        
        
        supports_bitsandbytes = self._check_bitsandbytes_support()
        
        hardware_info = HardwareInfo(
            has_cuda=has_cuda,
            has_mps=has_mps,
            cuda_device_count=cuda_device_count,
            gpu_names=gpu_names,
            gpu_memory_gb=gpu_memory_gb,
            total_ram_gb=memory.total / (1024**3),
            available_ram_gb=memory.available / (1024**3),
            cpu_count=psutil.cpu_count(),
            supports_bitsandbytes=supports_bitsandbytes
        )

        print(f"=====SYSTEM CONFIGURAITONS======")
        print(f"CONFIG: CPU: {hardware_info.cpu_count} cores")
        print(f"CONFIG: RAM: {hardware_info.total_ram_gb:.1f}GB total | {hardware_info.available_ram_gb:.1f}GB available")

        
        if hardware_info.has_cuda:
            print(f"CONFIG: GPU: ✅ NVIDIA CUDA Support")
            for i, (name, memory) in enumerate(zip(hardware_info.gpu_names, hardware_info.gpu_memory_gb)):
                print(f"CONFIG: GPU {i}: {name} ({memory:.1f}GB VRAM)")
        elif hardware_info.has_mps:
            print(f"CONFIG: GPU: ✅ Apple Silicon (MPS) Support")
            print(f"        {hardware_info.gpu_names[0]}")
        else:
            print("CONFIG: GPU: ❌ No GPU acceleration available")
        
        
        if hardware_info.supports_bitsandbytes:
            print("INFO:   Quantization: 4-bit/8-bit quantization AVAILABLE")
        else:
            print("INFO:   Quantization: Limited to CPU-based approaches (NOT AVIALABLE)")
        
        
        disk_usage = psutil.disk_usage('.')
        free_space_gb = disk_usage.free / (1024**3)
        print(f"CONFIG: Disk: {free_space_gb:.1f}GB free space")
        
        if free_space_gb < 40:
            print("Warning: Low disk space! Models require ~34GB total")
        
        self.hardware_info = hardware_info
        return hardware_info

    def get_compatible_quantization_options(self, model: ModelConfig) -> List[Dict]:
        """Get quantization options compatible with current hardware for the given model"""
        hw_info = self.hardware_info
        available_ram = hw_info.available_ram_gb
        compatible_options = []
        
        for quant in self.quantization_options:
            if quant.requires_cuda and not hw_info.has_cuda:
                continue
            if quant.requires_gpu and not (hw_info.has_cuda or hw_info.has_mps):
                continue
                        
            base_memory_req = model.size_gb * quant.memory_multiplier
            safety_factor = 1.5  # 50% overhead for loading and inference
            total_memory_req = base_memory_req * safety_factor
            
            # Check if system can handle this configuration
            if available_ram >= total_memory_req:
                compatible_options.append({
                    "quantization": quant.name,
                    "config": quant.config,
                    "fallback_dtype": quant.fallback_dtype,
                    "required_ram": total_memory_req,
                    "description": quant.description,
                    "requires_cuda": quant.requires_cuda,
                    "requires_gpu": quant.requires_gpu
                })
        
        return compatible_options
    
    def download_and_load_model(self, model: ModelConfig, option: Dict) -> Tuple[Optional[object], Optional[object], Dict]:
        """Load model with a specific quantization option"""
        print(f"PROCESSING: Loading with {option['quantization']}...")
        print(f"INFO: Expected RAM usage: {option['required_ram']:.1f}GB")
        
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

            try:
                if hasattr(model_obj, 'device'):
                    device = str(model_obj.device)
                elif hasattr(model_obj, 'hf_device_map'):
                    device = str(model_obj.hf_device_map)
                else:
                    # Try to infer from first parameter
                    device = str(next(model_obj.parameters()).device)
            except:
                device = "unknown"
            
            return model_obj, tokenizer, {
                "quantization": option["quantization"],
                "load_time": load_time,
                "device": device,
                "error": None
            }
            
        except Exception as e:
            print(f"     ❌ Failed: {str(e)[:100]}...")
            return None, None, {"error": str(e)}

    def _load_tokenizer_safely(self, model_id: str):
        """Load tokenizer with comprehensive error handling"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True  # Use fast tokenizer when available
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            print(f"     ❌ Tokenizer loading failed: {str(e)[:100]}...")
            return None

    def _load_model_with_strategy(self, model_id: str, option: Dict, hw_info: HardwareInfo):
        """Load model using the most appropriate strategy for the hardware"""
        
        # Determine optimal loading parameters
        if hw_info.has_cuda:
            if hw_info.cuda_device_count > 1:
                device_map = "auto"  
            else:
                device_map = "cuda:0"  
        elif hw_info.has_mps:
            device_map = "mps"
        else:
            device_map = "cpu"


        torch_dtype = option.get('fallback_dtype', torch.float16 if hw_info.has_cuda else torch.float32)
        
        loading_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch_dtype,
        }
        
       
        if option['config'] is not None:
            loading_kwargs["quantization_config"] = option['config']
        
       
        if device_map:
            loading_kwargs["device_map"] = device_map
        
       
        if not hw_info.has_cuda and not hw_info.has_mps:
            loading_kwargs.update({
                "torch_dtype": torch.float32,  # Force float32 for CPU
                "device_map": "cpu",
                "offload_folder": "./offload_tmp"  # Enable disk offloading if needed
            })
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **loading_kwargs)
            return model
        except Exception as e:
            
            print(f"OOOOPS, GOT ERROR:  Primary strategy failed, trying fallback...")
            return self._load_model_fallback(model_id, hw_info)

    def _load_model_fallback(self, model_id: str, hw_info: HardwareInfo):
        """Fallback loading strategy for difficult cases"""
        try:
            # Most conservative approach
            fallback_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float32,
                "device_map": "cpu"
            }
            
            model = AutoModelForCausalLM.from_pretrained(model_id, **fallback_kwargs)
            
            # Try to move to GPU after loading if available
            if hw_info.has_cuda:
                try:
                    model = model.to("cuda")
                    model = model.half()  
                except Exception:
                    pass  
            elif hw_info.has_mps:
                try:
                    model = model.to("mps")
                except Exception:
                    pass  
            
            return model
            
        except Exception as e:
            print(f"OOOOPS, GOT ERROR:   ❌ Fallback strategy also failed: {str(e)[:100]}...")
            return None

    def _verify_model_functionality(self, model, tokenizer) -> bool:
        """Verify that the loaded model actually works"""
        try:
            # Quick inference test
            test_input = tokenizer("Hello World!", return_tensors="pt")

            try:
                if hasattr(model, 'device'):
                    device = str(model.device)
                elif hasattr(model, 'hf_device_map'):
                    device = str(model.hf_device_map)
                else:
                    # Try to infer from first parameter
                    device = str(next(model.parameters()).device)
            except:
                device = "unknown"

            if device != "cpu" and device != "unknown":
                test_input = {k: v.to(device) for k, v in test_input.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    test_input["input_ids"],
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                print("Testing done!!", outputs)
            
            return True
            
        except Exception as e:
            print(f"OOOOPS, GOT ERROR:   Functionality test failed: {str(e)[:100]}...")
            return False

    def _cleanup_failed_model(self, model):
        """Clean up a failed model to free memory"""
        try:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    def _monitor_system_safely(self):
        """Safe system monitoring during model loading"""
        try:
            for i in range(60):  
                try:
                    memory = psutil.virtual_memory()
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    
                    status = f"     RAM: {memory.percent:.1f}% | CPU: {cpu_percent:.1f}%"
                    
                   
                    if self.hardware_info.has_cuda:
                        try:
                            gpu_mem = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                            status += f" | GPU: {gpu_mem:.1f}%"
                        except:
                            pass
                    
                    print(f"\r{status:<80}", end='', flush=True)
                    time.sleep(2)
                    
                    if memory.percent > 95:
                        print(f"\n    WARNING: Critical memory usage: {memory.percent:.1f}%")
                        break
                        
                except Exception:
                    time.sleep(2)
                    continue
                    
        except Exception:
            pass
        finally:
            print(f"\r{' ' * 80}\r", end='', flush=True)

    def display_model_options(self):
        "UI"

        print("\n Available Models:")
        print("-" * 90)
        print(f"{'Model':<15} {'Size':<8} {'Min RAM':<10} {'Rec RAM':<10} {'Best Option':<15} {'Description'}")
        print("-" * 90)
        
        for i, model in enumerate(self.available_models, 1):
            compatible_options = self.get_compatible_quantization_options(model)
            
            if compatible_options:
                best_option = compatible_options[0]["quantization"]
                status = "COMPATIBLE"
            else:
                best_option = "None"
                status = "NOT compatible"
            
            print(f"{i}. {model.name:<12} {model.size_gb:<6.1f}GB {model.min_ram_gb:<8.1f}GB {model.recommended_ram_gb:<8.1f}GB {best_option:<13} {model.description} {status}")

    def show_detailed_compatibility(self, models: List[ModelConfig]):
        """compatibility analysis with hardware-specific details"""
        print("\n Hardware-Specific Compatibility Analysis:")
        print("=" * 70)
        
        for model in models:
            print(f"\n {model.name} ({model.size_gb}GB)")
            compatible_options = self.get_compatible_quantization_options(model)
            
            if compatible_options:
                print("  Compatible configurations:")
                for i, option in enumerate(compatible_options, 1):
                    gpu_req = " GPU Required" if option["requires_gpu"] else " CPU Compatible"
                    cuda_req = " CUDA Required" if option["requires_cuda"] else ""
                    
                    print(f"      {i}. {option['quantization']:<12} | {option['required_ram']:.1f}GB RAM | {gpu_req} {cuda_req}")
                    print(f"         {option['description']}")
            else:
                print("    No compatible configurations for your hardware")
                print("    Consider: upgrading RAM, enabling GPU, or trying smaller models")

    def run_interactive_benchmark(self):
        """main interactive benchmarking flow"""
        print("   Interactive LLM Local Benchmarking Tool")
        print("   Supports: CPU-only, CUDA, Apple Silicon (MPS), Multi-GPU")
        print("=" * 60)
        
        # Step 1: get hardware system information
        self.get_hardware_info()
        
        # Step 2: UI for user to select the model
        self.display_model_options()
        selected_models = self.get_user_model_selection()
        
        if not selected_models:
            print("OOOOPS, No models selected. Exiting.")
            return
        
        # Once selected, it will show the available quantized versions
        self.show_detailed_compatibility(selected_models)
        
        # User Confirmation
        if not self.confirm_proceed():
            return
        
        # Benchmarking
        print(f"\n Starting hardware-optimized benchmark of {len(selected_models)} model(s)...")
        
        for model_config in selected_models:
            compatible_options = self.get_compatible_quantization_options(model_config)
            
            for option in compatible_options:
                print(f"\n--- Testing {model_config.name} with {option['quantization']} ---")
                model, tokenizer, model_info = self.download_and_load_model(model_config, option)
                
                if model is None:
                    self.results.append({
                        "model_name": f"{model_config.name}-{option['quantization']}",
                        "status": "failed",
                        "error": model_info.get("error", "Unknown error"),
                        "hardware_type": self._get_hardware_type()
                    })
                    continue
                
                result = self.advanced_benchmark(model, tokenizer, f"{model_config.name}-{option['quantization']}", model_info)
                result["hardware_type"] = self._get_hardware_type()
                self.results.append(result)
                
                self.cleanup_model(model, tokenizer)
            
        # Step 6: Hardware-Aware Analysis
        self.analyze_and_recommend()

    def _get_hardware_type(self) -> str:
        """Get descriptive hardware type for results"""

        hw = self.hardware_info
        if hw.has_cuda and hw.cuda_device_count > 1:
            return f"Multi-GPU CUDA ({hw.cuda_device_count} GPUs)"
        elif hw.has_cuda:
            return "Single GPU CUDA"
        elif hw.has_mps:
            return "Apple Silicon (MPS)"
        else:
            return "CPU-only"

    def get_user_model_selection(self) -> List[ModelConfig]:
        """Model selection with better guidance"""
        while True:
            try:
                print(f"\n Recommendation based on your {self._get_hardware_type()} system:")
                
                # Provide hardware-specific recommendations
                if self.hardware_info.has_cuda:
                    print("INFO:    You can run models with quantization for best performance")
                elif self.hardware_info.has_mps:
                    print("INFO:    Apple Silicon optimized - medium models recommended")
                else:
                    print("INFO:    CPU-only detected - start with smaller models")
                
                selection = input("\n Select models to benchmark (e.g., '1,3' or 'all'): ").strip().lower()
                
                if selection == 'all':
                    # Filter to only compatible models
                    compatible_models = []
                    for model in self.available_models:
                        if self.get_compatible_quantization_options(model):
                            compatible_models.append(model)
                    return compatible_models
                
                if selection in ['exit', 'quit']:
                    print("Exiting...")
                    exit(0)
                
                # Parse selections
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_models = []
                
                for idx in indices:
                    if 0 <= idx < len(self.available_models):
                        model = self.available_models[idx]
                        compatible_options = self.get_compatible_quantization_options(model)
                        
                        if compatible_options:
                            selected_models.append(model)
                        else:
                            print(f"OOPS, {model.name} cannot run on your {self._get_hardware_type()} system")
                    else:
                        print(f"OOPS, Invalid selection: {idx + 1}")
                
                if selected_models:
                    return selected_models
                else:
                    print("OOPS, No valid models selected. Please try again.")
                    
            except (ValueError, KeyboardInterrupt):
                print("OOPS, Invalid input. Please enter numbers separated by commas or 'all'")

    def confirm_proceed(self) -> bool:
        """Ask user confirmation to proceed with information"""
        print("="*50)
        print(f"\n Summary:")
        print(f"   Hardware: {self._get_hardware_type()}")
        print(f"   Quantization Support: {' Full (4-bit/8-bit)' if self.hardware_info.supports_bitsandbytes else ' Limited (CPU-only)'}")
        
        while True:
            response = input("\n Proceed with benchmarking? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                print("OOPS, Benchmarking cancelled.")
                return False
            else:
                print("Please enter 'y' or 'n'")

    def advanced_benchmark(self, model, tokenizer, model_name: str, model_info: Dict) -> Dict:
        """Benchmarking with hardware-specific optimizations"""
        print(f"\n⚡ Running hardware-optimized benchmark for {model_name}...")
        
        results = {
            "model_name": model_name,
            "quantization": model_info["quantization"],
            "load_time": model_info["load_time"],
            "device": model_info.get("device", "unknown")
        }
        
        try:
            # Test different prompt lengths
            for prompt_type, prompt_text in self.test_prompts.items():
                print(f" Testing {prompt_type} prompt...")
                
                # Multiple runs for statistical accuracy
                times = []
                # first_token_times = []
                tokens_per_second = []
                
                for run in range(3):  # 3 runs for averaging
                    metrics = self._benchmark_single_run(model, tokenizer, prompt_text)
                    if metrics:
                        times.append(metrics["total_time"])
                        # first_token_times.append(metrics["first_token_time"])
                        tokens_per_second.append(metrics["tokens_per_second"])
                
                if times:  # If we have successful runs
                    # Calculate statistics
                    avg_total_time = sum(times) / len(times)
                    # avg_first_token_time = sum(first_token_times) / len(first_token_times)
                    avg_tokens_per_second = sum(tokens_per_second) / len(tokens_per_second)
                    
                    # Calculate tokens per minute
                    tpm = avg_tokens_per_second * 60
                    
                    results[f"{prompt_type}_tpm"] = round(tpm, 2)
                    results[f"{prompt_type}_tps"] = round(avg_tokens_per_second, 2)
                    results[f"{prompt_type}_total_latency"] = round(avg_total_time, 3)
                    # results[f"{prompt_type}_first_token_latency"] = round(avg_first_token_time, 3)
                    
                    print(f"      {prompt_type.capitalize()}: {tpm:.1f} TPM | {avg_tokens_per_second:.1f} TPS")
                else:
                    print(f"OOPS:      {prompt_type.capitalize()}: Failed to benchmark")
            
            # memory usage tracking
            memory_stats = self._get_detailed_memory_stats()
            results.update(memory_stats)
            results["status"] = "success"
            
            return results
            
        except Exception as e:
            print(f"OOPS: Benchmark failed: {str(e)}")
            return {
                "model_name": model_name,
                "status": "failed",
                "error": str(e),
                "device": model_info.get("device", "unknown")
            }


    def _benchmark_single_run(self, model, tokenizer, prompt: str) -> Optional[Dict]:
        """single run benchmark with better timing"""
        try:
            
            try:
                if hasattr(model, 'device'):
                    device = str(model.device)
                elif hasattr(model, 'hf_device_map'):
                    device = str(model.hf_device_map)
                else:
                    # Try to infer from first parameter
                    device = str(next(model.parameters()).device)
            except:
                device = "unknown"
            
            # Prepare inputs
            inputs = tokenizer(prompt, return_tensors="pt")
            if device not in ["cpu", "unknown"]:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Measure generation time
            if self.hardware_info.has_cuda:
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            if self.hardware_info.has_cuda:
                torch.cuda.synchronize()
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            generated_tokens = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
            tokens_per_second = generated_tokens / total_time if total_time > 0 else 0
            
            # Estimate first token time (typically 10-30% of total time)
            # first_token_time = total_time * 0.15
            
            return {
                "total_time": total_time,
                # "first_token_time": first_token_time,
                "tokens_per_second": tokens_per_second,
                "generated_tokens": generated_tokens
            }
            
        except Exception as e:
            print(f"OOOOPS, GOT ERROR:      Single run failed: {str(e)[:50]}...")
            return None

    def _get_detailed_memory_stats(self) -> Dict:
        """Get comprehensive memory usage statistics"""
        stats = {}
        
        # RAM usage
        memory = psutil.virtual_memory()
        stats["ram_used_gb"] = round(memory.used / (1024**3), 2)
        stats["ram_percent"] = round(memory.percent, 1)
        
        # GPU memory if available
        if self.hardware_info.has_cuda:
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                stats.update({
                    "gpu_memory_allocated_gb": round(gpu_memory_allocated, 2),
                    "gpu_memory_reserved_gb": round(gpu_memory_reserved, 2),
                    "gpu_memory_total_gb": round(gpu_memory_total, 2),
                    "gpu_memory_percent": round((gpu_memory_allocated / gpu_memory_total) * 100, 1)
                })
            except Exception:
                stats["gpu_memory_gb"] = 0
        else:
            stats["gpu_memory_gb"] = 0
        
        return stats

    def cleanup_model(self, model, tokenizer):
        """cleanup with device-specific optimizations"""
        try:
            del model
            del tokenizer
            gc.collect()
            
            # Device-specific cleanup
            if self.hardware_info.has_cuda:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.hardware_info.has_mps:
                try:
                    torch.mps.empty_cache()
                except AttributeError:
                    pass  
            
            print("INFO:   Memory cleaned up")
            
        except Exception as e:
            print(f"OOOOPS, GOT ERROR:   Cleanup warning: {str(e)}")

    def analyze_and_recommend(self):
        """analysis with hardware-specific recommendations"""
        if not self.results:
            print(" No successful benchmarks to analyze")
            return
        
        successful_results = [r for r in self.results if r.get("status") == "success"]

        
        if not successful_results:
            print(" No successful benchmarks completed")
            return
        
        print("\n" + "=" * 70)
        print(" BENCHMARK RESULTS & HARDWARE ANALYSIS")
        print("=" * 70)
        model_groups = {}
        for result in successful_results:
            base_name = result['model_name'].split('-')[0] 
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append(result)
        

        for base_model, variants in model_groups.items():
            print(f"\n {base_model} - {len(variants)} configurations tested:")
            for variant in variants:
                quant_type = variant['model_name'].split('-', 1)[1]  
                print(f"    {quant_type}: {variant.get('medium_tpm', 0):.1f} TPM")
                

                
        for result in successful_results:
            print(f"\n {result['model_name']} ({result['quantization']})")
            print(f"   Device: {result.get('device', 'unknown')}")
            print(f"   Load Time: {result['load_time']:.1f}s")
            
            # Memory usage with hardware context
            if 'gpu_memory_allocated_gb' in result:
                print(f"   Memory: {result['ram_used_gb']:.1f}GB RAM ({result['ram_percent']:.1f}%)")
                print(f"           {result['gpu_memory_allocated_gb']:.1f}GB GPU ({result['gpu_memory_percent']:.1f}%)")
            else:
                print(f"   Memory: {result['ram_used_gb']:.1f}GB RAM ({result['ram_percent']:.1f}%)")
            
            print("   Performance:")
            for prompt_type in ['short', 'medium', 'long']:
                if f"{prompt_type}_tpm" in result:
                    tpm = result[f"{prompt_type}_tpm"]
                    tps = result[f"{prompt_type}_tps"]
                    # first_token = result[f"{prompt_type}_first_token_latency"] * 1000
                    total_latency = result[f"{prompt_type}_total_latency"]
                    print(f"     {prompt_type.capitalize():<8}: {tpm:6.1f} TPM | {tps:5.1f} TPS | {total_latency:.2f}s total")
        
        
        self._provide_hardware_recommendations(successful_results)
        
        
        self.save_detailed_results()

    def _provide_hardware_recommendations(self, results: List[Dict]):
        """Provide hardware-specific recommendations"""
        print(f"\n HARDWARE-OPTIMIZED RECOMMENDATIONS")
        print("-" * 45)
        
        # Find best model
        best_model = max(results, key=lambda x: x.get('medium_tpm', 0))
        
        print(f"   Best Overall: {best_model['model_name']} ({best_model['quantization']})")
        print(f"   Performance: {best_model['medium_tpm']:.1f} TPM")
        print(f"   Hardware Utilization: {best_model.get('device', 'unknown')}")
        
        
        hw_type = self._get_hardware_type()
        print(f"\n Optimization Tips for {hw_type}:")
        
        if self.hardware_info.has_cuda:
            if self.hardware_info.cuda_device_count > 1:
                print("    Multi-GPU detected - consider larger models with model parallelism")
                print("    4-bit quantization recommended for maximum throughput")
            else:
                print("    Single GPU - 4-bit quantization provides best memory efficiency")
                print("    Consider upgrading to higher VRAM GPU for larger models")
        elif self.hardware_info.has_mps:
            print("    Apple Silicon optimized - float16 models work well")
            print("    Unified memory architecture - larger models feasible")
        else:
            print("    CPU-only - focus on smaller, efficient models")
            print("    Consider adding GPU acceleration for significant speed improvements")
        
       
        if any(r.get('ram_percent', 0) > 80 for r in results):
            print("      High RAM usage detected - consider:")
            print("      • More aggressive quantization")
            print("      • Adding more system RAM")

    def save_detailed_results(self):
        """Save comprehensive results with hardware information"""
       
        for result in self.results:
            result.update({
                "hardware_type": self._get_hardware_type(),
                "has_cuda": self.hardware_info.has_cuda,
                "has_mps": self.hardware_info.has_mps,
                "cuda_device_count": self.hardware_info.cuda_device_count,
                "total_ram_gb": self.hardware_info.total_ram_gb,
                "supports_bitsandbytes": self.hardware_info.supports_bitsandbytes
            })
        
        df = pd.DataFrame(self.results)
        timestamp = int(time.time())
        filename = f"llm_benchmark_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n Results saved to: {filename}")
        
        # Also save a hardware summary
        hw_summary = {
            "benchmark_timestamp": timestamp,
            "hardware_type": self._get_hardware_type(),
            "total_ram_gb": self.hardware_info.total_ram_gb,
            "has_cuda": self.hardware_info.has_cuda,
            "cuda_device_count": self.hardware_info.cuda_device_count,
            "gpu_names": self.hardware_info.gpu_names,
            "supports_bitsandbytes": self.hardware_info.supports_bitsandbytes
        }
        
        hw_df = pd.DataFrame([hw_summary])
        hw_filename = f"hardware_info_{timestamp}.csv"
        hw_df.to_csv(hw_filename, index=False)
        print(f"Hardware info saved to: {hw_filename}")


if __name__ == "__main__":
    
    print(" Initializing LLM Benchmark Tool...")
    print("   Checking system compatibility...")
    
    try:
        benchmark = InteractiveLLMBenchmark()
        benchmark.run_interactive_benchmark()
    except KeyboardInterrupt:
        print("\n\n Benchmark interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {str(e)}")
        print("Please check your Python environment and dependencies.")
