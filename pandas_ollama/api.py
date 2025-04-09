"""
Ollama API ile iletişim kuran fonksiyonlar
"""

import requests
import json
import time
from typing import Dict, Optional

class OllamaClient:
    """Ollama API'si ile iletişim kurmak için istemci sınıfı"""
    
    def __init__(self, api_base: str = "http://localhost:11434", 
                model: str = "qwen2.5-7b-32k:latest",
                timeout: int = 60):
        self.api_base = api_base
        self.model = model
        self.api_url = f"{api_base}/api/chat"
        self.timeout = timeout
    
    def call_api(self, payload: Dict) -> str:
        """API'yi çağırır ve yanıtı döndürür"""
        try:
            # İlerleme göstergesi
            start_time = time.time()
            
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            try:
                result = response.json()
                # Yeni API response formatına uygun şekilde yanıtı al
                content = result.get("message", {}).get("content", "")
                
                elapsed = time.time() - start_time
                print(f"✓ API yanıtı alındı ({elapsed:.2f} saniye)")
                
                return content
            except json.JSONDecodeError:
                # JSON parse edilemezse, alternatif apileri dene
                print("⚠️ API yanıtı JSON formatında değil, alternatif endpointler deneniyor...")
                result = self.try_alternative_api(payload)
                if result:
                    elapsed = time.time() - start_time
                    print(f"✓ Alternatif API yanıtı alındı ({elapsed:.2f} saniye)")
                    return result
                else:
                    return f"API yanıtı işlenemedi: {response.text[:100]}..."
        except requests.exceptions.Timeout:
            print(f"⚠️ API isteği zaman aşımına uğradı ({self.timeout} saniye)")
            return "Ollama API isteği zaman aşımına uğradı. Lütfen daha sonra tekrar deneyin veya timeout süresini artırın."
        except requests.exceptions.RequestException as e:
            # 404 hatası için alternatif API'yi dene
            if "404" in str(e):
                print("⚠️ 404 hatası, alternatif endpointler deneniyor...")
                result = self.try_alternative_api(payload)
                if result:
                    return result
            raise e

    def try_alternative_api(self, payload: Dict) -> Optional[str]:
        """Alternatif API endpointlerini dener"""
        # Önce completions api'yi dene
        try:
            api_url = f"{self.api_base}/api/completions"
            
            # Payload'u completions formatına dönüştür
            if "messages" in payload:
                system_msg = next((m for m in payload["messages"] if m["role"] == "system"), None)
                user_msg = next((m for m in payload["messages"] if m["role"] == "user"), None)
                
                if system_msg and user_msg:
                    prompt = f"{system_msg['content']}\n\n{user_msg['content']}"
                else:
                    prompt = "\n\n".join([m["content"] for m in payload["messages"]])
            else:
                prompt = payload.get("prompt", "")
            
            completion_payload = {
                "model": payload.get("model", self.model),
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(api_url, json=completion_payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"⚠️ completions API hatası: {str(e)}")
            # Sonra generate api'yi dene
            try:
                api_url = f"{self.api_base}/api/generate"
                response = requests.post(api_url, json=completion_payload, timeout=self.timeout)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
            except Exception as e2:
                print(f"⚠️ generate API hatası: {str(e2)}")
                return None
    
    def ping(self) -> bool:
        """Ollama API'sinin çalışıp çalışmadığını kontrol eder"""
        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
