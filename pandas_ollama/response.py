"""
AI yanıtları için yapılandırılmış yanıt sınıfı
"""

import json
from typing import Dict, Any, Optional, List

class StructuredResponse:
    """
    MyPandasAI fonksiyonlarından dönen yapılandırılmış yanıt sınıfı.
    İçerik, hata mesajları, görselleştirme ve metadata gibi yanıt verilerini içerir.
    """
    
    def __init__(self, 
                content: str = "", 
                error: Optional[str] = None,
                visualization: Optional[str] = None,
                code: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Yapılandırılmış yanıt nesnesini başlatır.
        
        Args:
            content (str): Ana yanıt içeriği
            error (Optional[str]): Hata mesajı
            visualization (Optional[str]): Base64 kodlanmış görsel verisi
            code (Optional[str]): Yanıt tarafından üretilen kod
            metadata (Optional[Dict[str, Any]]): Ek meta veriler
        """
        self.content = content
        self.error = error
        self.visualization = visualization
        self.code = code
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Yanıtı sözlük olarak döndürür"""
        result = {
            "content": self.content,
            "metadata": self.metadata
        }
        
        if self.error:
            result["error"] = self.error
            
        if self.visualization:
            result["visualization"] = self.visualization
            
        if self.code:
            result["code"] = self.code
            
        return result
    
    def to_json(self) -> str:
        """Yanıtı JSON formatında döndürür"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def has_visualization(self) -> bool:
        """Görselleştirme verisi olup olmadığını kontrol eder"""
        return self.visualization is not None and len(self.visualization) > 0
        
    def __str__(self) -> str:
        """Okunabilir string temsilini döndürür"""
        result = []
        
        if self.content:
            result.append(f"Content: {self.content}")
            
        if self.error:
            result.append(f"Error: {self.error}")
            
        if self.visualization:
            result.append(f"Visualization: [Base64 data, length: {len(self.visualization)}]")
            
        if self.code:
            result.append(f"Code: {self.code[:50]}...")
            
        if self.metadata:
            result.append(f"Metadata: {self.metadata}")
            
        return "\n".join(result)
