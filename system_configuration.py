import psutil
import torch

def get_system_info():
        info = {
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "has_gpu": torch.cuda.is_available(),
            "gpu_memory_gb": 0
        }
        
        if info["has_gpu"]:
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu_name"] = torch.cuda.get_device_name(0)
        
        return info


if __name__ == "__main__":
      print(get_system_info())