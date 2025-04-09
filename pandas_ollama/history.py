"""
Analiz geçmişini yönetmek için modül
"""

import os
import json
from typing import Dict, List
from .response import StructuredResponse
from datetime import datetime

class HistoryManager:
    """Analiz geçmişi yönetim sınıfı"""
    
    def __init__(self, history_path: str = None, save_history: bool = False):
        """
        Args:
            history_path (str, optional): Geçmiş kayıt dosya yolu
            save_history (bool): Geçmişin kaydedilip kaydedilmeyeceği
        """
        self.history_path = history_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                       "../pandas_ai_history.json")
        self.save_history = save_history
        self.history = self._load_history() if save_history else []
    
    def _load_history(self) -> List[Dict]:
        """Analiz geçmişini yükler"""
        if not self.save_history:
            return []
            
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_history(self) -> None:
        """Analiz geçmişini kaydeder"""
        if not self.save_history:
            return
            
        try:
            # Eğer klasör yoksa oluştur
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Geçmiş kaydedilirken hata oluştu: {str(e)}")
    
    def add_to_history(self, query: str, result: StructuredResponse) -> None:
        """Yeni bir analizi geçmişe ekler"""
        if not self.save_history:
            return
            
        entry = {
            "query": query,
            "result": result.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)
        self._save_history()
    
    def get_history(self) -> List[Dict]:
        """Analiz geçmişini döndürür"""
        return self.history
    
    def clear_history(self) -> None:
        """Analiz geçmişini temizler"""
        self.history = []
        if self.save_history:
            self._save_history()
