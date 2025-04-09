"""
Ollama API ve PandasAI yapılandırma ayarları
"""

import os
import json
import time
from typing import Dict, Any, Optional

class OllamaConfig:
    """Ollama API yapılandırma ve optimizasyon yöneticisi"""
    
    DEFAULT_CONFIG = {
        "timeout": 60,           # API çağrıları için zaman aşımı (saniye)
        "max_rows": 100,         # Context için max satır sayısı
        "cache_size": 50,        # Önbellek büyüklüğü
        "temperature": 0.7,      # Yanıt çeşitliliği
        "max_tokens": 1024,      # Maksimum token sayısı
        "top_p": 0.9,            # Top-p örnekleme  
        "parallel_requests": False,  # Paralel istek gönderimi
        "verbose": True,         # Ayrıntılı çıktı
        "direct_execution": True, # LLM kodunu doğrudan çalıştırma
        "model": "qwen2.5:7b" # Varsayılan model
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Ollama yapılandırmasını başlatır.
        
        Args:
            config_path (str, optional): Yapılandırma dosyası yolu
        """
        # Yapılandırma dosyası, pandas_ollama klasörü içinde olmalı
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path or os.path.join(module_dir, "ollama_config.json")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Yapılandırmayı yükler veya varsayılan yapılandırmayı döndürür"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Varsayılan yapılandırmayı, yüklenen yapılandırma ile birleştir
                    return {**self.DEFAULT_CONFIG, **config}
            except:
                print(f"Yapılandırma dosyası yüklenirken hata oluştu: {self.config_path}")
                print("Varsayılan yapılandırma kullanılıyor.")
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self) -> None:
        """Mevcut yapılandırmayı dosyaya kaydeder"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            print(f"Yapılandırma kaydedildi: {self.config_path}")
        except Exception as e:
            print(f"Yapılandırma kaydedilirken hata oluştu: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Yapılandırma değerini döndürür"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Yapılandırma değerini ayarlar"""
        self.config[key] = value
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """Yapılandırmayı günceller"""
        self.config.update(new_config)
    
    def optimize_for_speed(self) -> None:
        """Hız için yapılandırmayı optimize eder"""
        speed_config = {
            "timeout": 30,
            "max_rows": 50,
            "temperature": 0.3,
            "max_tokens": 512,
            "top_p": 0.8,
            "model": "gemma:2b" # Daha küçük ve hızlı model
        }
        self.update(speed_config)
        print("Yapılandırma hız için optimize edildi.")
    
    def optimize_for_quality(self) -> None:
        """Kalite için yapılandırmayı optimize eder"""
        quality_config = {
            "timeout": 120,
            "max_rows": 200,
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.95,
            "model": "llama3:latest" # Daha büyük ve kapsamlı model
        }
        self.update(quality_config)
        print("Yapılandırma kalite için optimize edildi.")
    
    def __str__(self) -> str:
        """Yapılandırmayı string olarak döndürür"""
        return json.dumps(self.config, indent=2)
