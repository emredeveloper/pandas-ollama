"""
Google Colab'te Ollama API'yi kullanmak için adapter modülü
"""

import os
import time
import subprocess
import requests
from typing import Optional, Dict, Any

class OllamaColabAdapter:
    """
    Google Colab için Ollama adapter sınıfı
    Yerel Ollama sunucusunun olmaması durumunda SSH tünel üzerinden
    uzak bir Ollama sunucusuna bağlanmayı sağlar
    """
    
    def __init__(self, remote_host: Optional[str] = None, 
                remote_port: int = 11434,
                local_port: int = 11434,
                ssh_key_path: Optional[str] = None,
                timeout: int = 60):
        """
        Ollama Colab adapter'ını başlatır
        
        Args:
            remote_host: Uzak Ollama sunucusunun IP adresi veya host adı (ör: "example.com")
            remote_port: Uzak Ollama API port numarası
            local_port: Yerel port numarası (tünel için)
            ssh_key_path: SSH özel anahtar yolu
            timeout: Bağlantı zaman aşımı (saniye)
        """
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port
        self.ssh_key_path = ssh_key_path
        self.timeout = timeout
        self.tunnel_process = None
        self.is_colab = self._check_if_colab()
    
    def _check_if_colab(self) -> bool:
        """Google Colab ortamında çalışıp çalışmadığını kontrol eder"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def is_local_ollama_available(self) -> bool:
        """Yerel Ollama API'nin kullanılabilir olup olmadığını kontrol eder"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def setup_tunnel(self) -> bool:
        """SSH tünel bağlantısı kurar"""
        if not self.is_colab:
            print("Bu işlev yalnızca Google Colab'de çalışır.")
            return False
            
        if not self.remote_host:
            print("Uzak Ollama sunucusu belirtilmediği için tünel kurulamadı.")
            return False
            
        # SSH anahtar dosyası oluştur
        if self.ssh_key_path:
            key_path = self.ssh_key_path
        else:
            # Kullanıcıdan SSH anahtarını al
            from google.colab import files
            from IPython.display import display, HTML
            
            display(HTML("<p>SSH özel anahtarınızı yükleyin (id_rsa dosyası)</p>"))
            uploaded = files.upload()
            
            if not uploaded:
                print("SSH anahtarı yüklenmedi!")
                return False
                
            key_name = list(uploaded.keys())[0]
            key_path = f"/tmp/{key_name}"
            with open(key_path, "wb") as f:
                f.write(uploaded[key_name])
            os.chmod(key_path, 0o600)
        
        # SSH tüneli oluştur
        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-i", key_path,
            "-N", "-L", f"{self.local_port}:localhost:{self.remote_port}",
            f"root@{self.remote_host}"
        ]
        
        # Bazı durumlarda farklı kullanıcı adları ile çalışabilme esnekliği eklendi
        username = os.environ.get("SSH_USERNAME", "root")  # Varsayılan root, çevre değişkeni ile değiştirilebilir
        host_with_user = f"{username}@{self.remote_host}"
        
        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-i", key_path,
            "-N", "-L", f"{self.local_port}:localhost:{self.remote_port}",
            host_with_user
        ]
        
        try:
            self.tunnel_process = subprocess.Popen(cmd)
            time.sleep(2)  # Tünelin kurulmasını bekle
            
            # Tünelin çalışıp çalışmadığını kontrol et
            if self.is_local_ollama_available():
                print(f"✅ Ollama API tüneli başarıyla kuruldu! API şimdi localhost:{self.local_port} adresinde kullanılabilir")
                return True
            else:
                print("❌ Tünel kuruldu ancak Ollama API'ye erişilemiyor. Uzak sunucuda Ollama'nın çalıştığından emin olun.")
                return False
        except Exception as e:
            print(f"❌ Tünel kurulumunda hata: {str(e)}")
            return False
    
    def close_tunnel(self) -> None:
        """SSH tünelini kapatır"""
        if self.tunnel_process:
            self.tunnel_process.terminate()
            self.tunnel_process = None
            print("SSH tüneli kapatıldı.")
    
    def install_requirements(self) -> bool:
        """Gerekli kütüphaneleri kurar"""
        if not self.is_colab:
            return False
            
        print("Gerekli kütüphaneler kuruluyor...")
        
        try:
            # Gerekli paketleri kur
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib", "seaborn", "requests", "pandas", "numpy", "pillow"])
            
            return True
        except Exception as e:
            print(f"Kütüphane kurulumunda hata: {str(e)}")
            return False
    
    def setup_for_colab(self) -> str:
        """
        Colab ortamı için gerekli kurulumları yapar ve kullanım örneği döndürür
        
        Returns:
            str: Kullanım örnekleri
        """
        if not self.is_colab:
            return "Bu işlev yalnızca Google Colab'de çalışır."
        
        success = self.install_requirements()
        
        usage_example = """
# Ollama'yı Colab'de kullanma
from pandas_ollama import MyPandasAI
from pandas_ollama.colab_adapter import OllamaColabAdapter
import pandas as pd

# Örnek veriler
df = pd.DataFrame({
    'Product': ['Laptop', 'Phone', 'Tablet'],
    'Price': [1000, 800, 500],
    'Stock': [50, 100, 75]
})

# Uzak Ollama sunucusuna tünel oluştur
adapter = OllamaColabAdapter(remote_host="your-server-ip")
adapter.setup_tunnel()

# Pandas-Ollama'yı başlat
panoll = MyPandasAI(df, model="qwen2.5:7b")

# Veri analizi yap
result = panoll.ask("Bu verideki ortalama fiyat nedir?")
print(result.content)

# Görselleştirme oluştur
result = panoll.plot("Fiyatlara göre stok dağılımını göster", viz_type="scatter")

# Görselleştirmeyi görüntüle
if result.visualization:
    import base64
    from IPython.display import Image
    image_data = base64.b64decode(result.visualization)
    display(Image(data=image_data))

# İşiniz bittiğinde tüneli kapat
adapter.close_tunnel()
"""
        
        print("\n✅ Colab için gerekli kurulumlar tamamlandı!")
        return usage_example
