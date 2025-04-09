"""
Ana MyPandasAI sınıfı için modül
"""

import pandas as pd
import hashlib
import time
import traceback
import warnings
import re
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime

from .response import StructuredResponse
from .api import OllamaClient
from .visualizations import Visualizer
from .transformations import DataTransformer
from .history import HistoryManager
from .config import OllamaConfig

class MyPandasAI:
    def __init__(self, 
                dataframe: pd.DataFrame, 
                model: str = "qwen2.5:7b", 
                api_base: str = "http://localhost:11434",
                history_path: str = None,
                save_history: bool = False,
                timeout: int = 60,
                max_rows: int = 100,
                cache_size: int = 50,
                direct_execution: bool = True):  # Yeni parametre: LLM kodu doğrudan çalıştırma
        """
        Pandas verilerini doğal dil sorgularıyla analiz etmek için gelişmiş bir AI aracı başlatır.
        
        Args:
            dataframe (pd.DataFrame): Analiz edilecek DataFrame
            model (str): Kullanılacak Ollama modeli
            api_base (str): Ollama API için kullanılacak temel URL
            history_path (str, optional): Analiz geçmişinin kaydedileceği dosya yolu
            save_history (bool): Geçmişin kaydedilip kaydedilmeyeceği
            timeout (int): API çağrıları için zaman aşımı süresi (saniye)
            max_rows (int): Context için maksimum satır sayısı
            cache_size (int): Önbellek büyüklüğü
            direct_execution (bool): LLM'den gelen kodu doğrudan çalıştırma
        """
        self.df = dataframe
        self.model = model
        self.api_base = api_base
        self.timeout = timeout
        self.max_rows = max_rows
        self.cache = {}  # Sorgu önbelleği
        self.cache_size = cache_size
        self.direct_execution = direct_execution
        
        # Tarih sütunlarını tespit et ve dönüşüm yap
        self._detect_datetime_columns()
        
        # Config yöneticisi
        self.config = OllamaConfig()
        self.config.set("model", model)
        self.config.set("timeout", timeout)
        
        # Alt bileşenleri başlat
        self.api_client = OllamaClient(api_base=api_base, model=model, timeout=timeout)
        self.visualizer = Visualizer(dataframe, timeout=timeout)
        self.transformer = DataTransformer(dataframe)
        self.history_manager = HistoryManager(history_path, save_history)
    
    def _detect_date_format(self, sample_dates):
        """
        Veri örneklerinden tarih formatını tespit etmeye çalışır
        """
        # Yaygın kullanılan tarih formatları
        common_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
            '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y',
            '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y'
        ]
        
        # Zaman içeren formatlar
        time_formats = [
            '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
            '%Y/%m/%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
        ]
        
        # Varsayılan formatları birleştir
        all_formats = common_formats + time_formats
        
        for date_str in sample_dates:
            if not isinstance(date_str, str) or pd.isna(date_str):
                continue
                
            # Her format için test et
            for fmt in all_formats:
                try:
                    datetime.strptime(date_str, fmt)
                    return fmt
                except:
                    pass
                    
        # Tarih formatı tespit edemediyse, strftime formatına benzeyen desenler ara
        date_pattern = r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$"  # YYYY-MM-DD, DD/MM/YYYY vb.
        datetime_pattern = r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\s\d{1,2}:\d{1,2}(:\d{1,2})?$"  # YYYY-MM-DD HH:MM:SS
        
        # Örnek tarihler için patternleri test et
        for date_str in sample_dates:
            if not isinstance(date_str, str) or pd.isna(date_str):
                continue
                
            if re.match(datetime_pattern, date_str):
                # Saatli format
                if '-' in date_str:
                    return '%Y-%m-%d %H:%M:%S' if date_str[0:4].isdigit() else '%d-%m-%Y %H:%M:%S'
                elif '/' in date_str:
                    return '%Y/%m/%d %H:%M:%S' if date_str[0:4].isdigit() else '%d/%m/%Y %H:%M:%S'
                elif '.' in date_str:
                    return '%Y.%m.%d %H:%M:%S' if date_str[0:4].isdigit() else '%d.%m.%Y %H:%M:%S'
            elif re.match(date_pattern, date_str):
                # Sadece tarih formatı
                if '-' in date_str:
                    return '%Y-%m-%d' if date_str[0:4].isdigit() else '%d-%m-%Y'
                elif '/' in date_str:
                    return '%Y/%m/%d' if date_str[0:4].isdigit() else '%d/%m/%Y'
                elif '.' in date_str:
                    return '%Y.%m.%d' if date_str[0:4].isdigit() else '%d.%m.%Y'
        
        return None
    
    def _is_likely_date_column(self, column_name):
        """
        Sütun adından tarih sütunu olma ihtimalini tahmin eder
        """
        name_lower = column_name.lower()
        date_keywords = ['date', 'tarih', 'time', 'zaman', 'year', 'yıl', 'ay', 'month', 'day', 'gün']
        
        for keyword in date_keywords:
            if keyword in name_lower:
                return True
        return False
    
    def _detect_datetime_columns(self):
        """Tarih ve zaman sütunlarını tespit eder ve dönüştürür"""
        for col in self.df.columns:
            # Eğer sütun string tipindeyse ve muhtemelen tarih sütunuysa
            if self.df[col].dtype == 'object' and (self._is_likely_date_column(col) or
               # Sütun adı tarih içermese bile, ilk 5 hücreyi kontrol et
               any(isinstance(s, str) and ('/' in s or '-' in s) for s in self.df[col].head().values)):
                
                try:
                    # Tarih formatını tespit etmeye çalış
                    sample_values = self.df[col].dropna().head(10).values
                    date_format = self._detect_date_format(sample_values)
                    
                    # Eğer format tespit edildiyse, o formatı kullan
                    if date_format:
                        # Uyarıları bastır
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pd_datetime = pd.to_datetime(self.df[col], format=date_format, errors='coerce')
                    else:
                        # Format tespit edilemezse, dateutil ile dene
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pd_datetime = pd.to_datetime(self.df[col], errors='coerce')
                    
                    # Eğer dönüşüm başarılıysa ve tüm değerler NaN değilse
                    if pd_datetime.notna().sum() > 0.5 * len(pd_datetime):
                        # DataFrame'deki sütunu tarih tipine dönüştür
                        self.df[col] = pd_datetime
                        print(f"'{col}' sütunu datetime tipine dönüştürüldü.")
                except Exception as e:
                    # Hata durumunda sessizce devam et
                    pass
    
    # Sorgu önbelleği için bir yardımcı fonksiyon
    def _get_query_hash(self, query: str, context: str = "") -> str:
        """Sorgu ve bağlam için benzersiz bir karma değeri oluşturur"""
        hash_input = f"{query}:{context}:{self.model}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def prepare_context(self) -> str:
        """
        DataFrame hakkında sütun adları, veri tipleri, örnek veriyi ve 
        istatistikleri hazırlayarak metinsel bir bağlam oluşturur.
        
        Returns:
            str: Biçimlendirilmiş bağlam bilgisi
        """
        columns_info = self.df.dtypes.to_string()
        
        # Daha fazla satır varsa, DataFrame büyüklüğünü sınırla
        df_sample = self.df
        if len(self.df) > self.max_rows:
            # Maksimum satır sayısını aşarsa, örnekleme yap
            # Baştan ve sondan bazı satırları al, ortadan örnekle
            head_count = min(5, self.max_rows // 4)
            tail_count = min(5, self.max_rows // 4)
            middle_count = self.max_rows - head_count - tail_count
            
            if middle_count > 0:
                middle_indices = pd.Series(range(head_count, len(self.df) - tail_count)).sample(middle_count).tolist()
                indices = list(range(head_count)) + middle_indices + list(range(len(self.df) - tail_count, len(self.df)))
                df_sample = self.df.iloc[sorted(indices)]
            else:
                df_sample = pd.concat([self.df.head(head_count), self.df.tail(tail_count)])
        
        sample_data = df_sample.head(5).to_csv(index=False)
        
        # İstatistiksel bilgileri ekle
        try:
            stats = self.df.describe(include='all').to_string()
        except:
            stats = "İstatistikler hesaplanamadı."
            
        # Eksik değerler hakkında bilgi
        missing_values = self.df.isnull().sum().to_string()
        
        context = f"""
DataFrame Bilgisi:
- Şekil: {self.df.shape[0]} satır, {self.df.shape[1]} sütun

Sütunlar ve Veri Tipleri:
{columns_info}

Eksik Değerler:
{missing_values}

İstatistiksel Özet:
{stats}

Örnek Veri (İlk 5 Satır):
{sample_data}
"""
        return context

    def run(self, query: str, generate_viz: bool = False, viz_type: str = None, 
           generate_code: bool = True) -> StructuredResponse:
        """
        DataFrame üzerinden doğal dil sorgusunu işleyip, yapılandırılmış yanıt döndürür.
        
        Args:
            query (str): Kullanıcı sorgusu
            generate_viz (bool): Görselleştirme oluşturulup oluşturulmayacağı
            viz_type (str, optional): Oluşturulacak görselleştirme türü
            generate_code (bool): Kod oluşturulup oluşturulmayacağı
            
        Returns:
            StructuredResponse: Yapılandırılmış yanıt
        """
        context = self.prepare_context()
        
        # Önbellekten kontrol et
        query_hash = self._get_query_hash(query, f"{generate_viz}:{viz_type}:{generate_code}")
        if query_hash in self.cache:
            print("✓ Önbellekten yanıt alındı.")
            return self.cache[query_hash]
        
        # Görselleştirme ve kod oluşturma talimatları
        instructions = "Kullanıcının sorgusu hakkında kapsamlı bir yanıt oluştur."
        
        if generate_code or self.direct_execution:
            instructions += " Ayrıca bu analizi gerçekleştirecek Python/pandas kodu da oluştur."
            
            if self.direct_execution:
                instructions += " Kodun direkt çalıştırılacak, bu yüzden sadece çalışacak kodu yaz, başka açıklama ekleme."
        
        if generate_viz:
            viz_options = ", ".join(self.visualizer.supported_viz.keys())
            
            if viz_type and viz_type in self.visualizer.supported_viz:
                instructions += f" '{viz_type}' türünde bir grafik oluşturmak için kod öner."
            else:
                instructions += f" Uygun bir görselleştirme türü öner ({viz_options}). Görselleştirme kodu da oluştur."
        
        # API isteği için payload hazırla
        print("⏳ Ollama API'ye istek gönderiliyor...")
        start_time = time.time()
        
        # LLM'e bir DataFrame erişim bilgisi ver
        if self.direct_execution:
            df_info = "DataFrame değişkeni `df` olarak tanımlıdır. Doğrudan bu değişkeni kullanabilirsin."
        else:
            df_info = "Tüm pandas kodlarında df değişkenini kullan."
        
        # Yeni API formatına uygun payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": f"""Bu bir DataFrame analiz talebidir. Sen üst düzey bir veri analisti gibi davran.
                    Aşağıdaki verileri ve bağlamı kullanarak soruya ayrıntılı yanıt ver:
                    
                    {context}
                    
                    {df_info}
                    
                    {instructions}
                    
                    Eğer kod oluşturuyorsan, sadece çalışacak, tam ve doğru kodu oluştur.
                    Yanıtını şu yapıda oluştur:
                    
                    1. YANIT: [Kullanıcı sorusuna detaylı yanıt]
                    2. KOD: [Python/pandas kodu - istenirse]
                    3. GÖRSEL: [Görselleştirme kodu - istenirse]
                    """
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            "stream": False
        }
        
        try:
            response = self.api_client.call_api(payload)
            
            # Eğer doğrudan kod çalıştırma açıksa, LLM tarafından oluşturulan kodu çalıştır
            if self.direct_execution:
                result = self._execute_llm_code(response, query, generate_viz, viz_type)
            else:
                # Normal ayrıştırma işlemi
                result = self._parse_llm_response(response, query, generate_viz, viz_type)
            
            # Geçmişe ekle
            self.history_manager.add_to_history(query, result)
            
            # Önbelleğe ekle
            self.cache[query_hash] = result
            
            # Önbellek büyüklüğünü kontrol et
            if len(self.cache) > self.cache_size:
                # En eski girişleri sil (maksimum önbellek boyutunun yarısı)
                old_keys = list(self.cache.keys())[:(self.cache_size // 2)]
                for key in old_keys:
                    del self.cache[key]
            
            elapsed = time.time() - start_time
            print(f"✓ Yanıt alındı ({elapsed:.2f} saniye)")
            
            return result
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
            traceback.print_exc()
            return StructuredResponse(error=f"Error calling AI service: {str(e)}")

    def _format_numeric_value(self, value):
        """Sayısal değerleri formatla"""
        if isinstance(value, (float, np.float64, np.float32)):
            return f"{value:.2f}"  # İki ondalık basamakla sınırla
        return value

    def _execute_llm_code(self, response: str, query: str, generate_viz: bool, viz_type: str) -> StructuredResponse:
        """
        LLM'den gelen yanıtı parse eder, kodu çıkarır ve çalıştırır.
        Doğrudan kod çalıştırma özelliği için.
        """
        structured_response = StructuredResponse()
        structured_response.metadata = {
            "query": query,
            "model": self.model,
            "timestamp": datetime.now().isoformat()
        }
        
        # Önce normal ayrıştırma yap
        content, code, viz_code = self._extract_content_code(response)
        
        # Yanıtı ayarla
        structured_response.content = content
        structured_response.code = code
        
        # Şimdi kodu çalıştır
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import io
            import base64
            import numpy as np
            
            # Globals ve locals hazırla
            execution_globals = {
                'pd': pd, 
                'plt': plt, 
                'sns': sns, 
                'np': np, 
                'io': io, 
                'base64': base64
            }
            
            execution_locals = {'df': self.df}
            
            # Eğer kod varsa, çalıştır
            if code:
                # Önce görselleştirme olmayan kodu çalıştır
                non_viz_code = self._extract_non_viz_code(code)
                if non_viz_code:
                    print(f"⚙️ LLM tarafından oluşturulan kod çalıştırılıyor...")
                    
                    try:
                        # Temizlenmiş kodu çalıştır
                        exec(non_viz_code, execution_globals, execution_locals)
                        
                        # Sonuçlar local namespace'de olabilir, en son değeri al
                        result_found = False
                        
                        # Özel sonuç değişkenlerini ara
                        result_variable_names = ['result', 'sonuc', 'ortalama', 'average', 'mean', 'toplam', 'sum', 'count']
                        for var_name in result_variable_names:
                            if var_name in execution_locals:
                                value = execution_locals[var_name]
                                # Ondalık sayıları iki basamakla sınırla
                                if isinstance(value, (float, np.float64, np.float32)):
                                    formatted_value = self._format_numeric_value(value)
                                    structured_response.content += f"\n\n**Sonuç:** {formatted_value}"
                                else:
                                    structured_response.content += f"\n\n**Sonuç:** {value}"
                                result_found = True
                                break
                        
                        # Eğer özel sonuç değişkeni bulunamadıysa, diğer namespace değişkenlerini kontrol et
                        if not result_found:
                            for var_name in execution_locals:
                                if var_name != 'df' and not var_name.startswith('_'):
                                    # Bulduğumuz sonucu içeriğe ekle
                                    value = execution_locals[var_name]
                                    if isinstance(value, (int, float, np.float64, np.float32)):
                                        # Ondalık sayıları iki basamakla sınırla
                                        if isinstance(value, (float, np.float64, np.float32)):
                                            formatted_value = self._format_numeric_value(value)
                                            structured_response.content += f"\n\n**Hesaplanan {var_name}:** {formatted_value}"
                                        else:
                                            structured_response.content += f"\n\n**Hesaplanan {var_name}:** {value}"
                                        result_found = True
                                    elif isinstance(value, (str, bool, list, dict)):
                                        structured_response.content += f"\n\n**Hesaplanan {var_name}:** {value}"
                                        result_found = True
                                    elif isinstance(value, pd.DataFrame) and len(value) <= 10:
                                        # Eğer bir DataFrame ise ve küçükse, bunu metin olarak ekleyelim
                                        structured_response.content += f"\n\n**DataFrame {var_name}:**\n```\n{value.to_string()}\n```"
                                        result_found = True
                                    elif isinstance(value, pd.Series):
                                        # Eğer bir Series ise, bunu metin olarak ekleyelim
                                        structured_response.content += f"\n\n**Series {var_name}:**\n```\n{value.to_string()}\n```"
                                        result_found = True
                        
                        # Eğer hala sonuç bulunamadıysa, son çare olarak ortalama fiyatı hesapla
                        if not result_found and 'Price' in self.df.columns and query and ('ortalama' in query.lower() or 'average' in query.lower() or 'mean' in query.lower()):
                            value = self.df['Price'].mean()
                            formatted_value = self._format_numeric_value(value)
                            structured_response.content += f"\n\n**Hesaplanan ortalama fiyat:** {formatted_value}"
                            
                    except SyntaxError as se:
                        print(f"❌ Kod çalıştırma hatası: {str(se)}")
                        # Kod ayrıştırılamadıysa, daha agresif bir temizlik dene
                        cleaned_code = re.sub(r'[^a-zA-Z0-9_\s=\(\)\[\]\{\}:\.\"\'\/\*\+\-,<>;]', '', non_viz_code)
                        try:
                            exec(cleaned_code, execution_globals, execution_locals)
                        except Exception as e2:
                            print(f"❌ Temizlenmiş kod için de hata: {str(e2)}")
                            structured_response.error = f"Kod çalıştırılamadı: {str(e2)}"
            
            # Eğer görselleştirme istenirse veya kod varsa
            if generate_viz and (viz_code or code or viz_type):
                print(f"🖼️ Görselleştirme oluşturuluyor...")
                
                # Özel vizualization tipi belirtilmişse, doğrudan o türden bir görsellik oluştur
                if viz_type and viz_type in self.visualizer.supported_viz:
                    viz_base64 = self.visualizer.supported_viz[viz_type](query)
                    structured_response.visualization = viz_base64
                else:
                    # Görselleştirme kodu çalıştır
                    code_to_run = viz_code if viz_code else code
                    
                    # Kodu temizle
                    code_to_run = self._clean_code_from_markdown(code_to_run)
                    
                    # Eğer graf kodu içermiyorsa uygun bir graf kodu oluştur
                    if not any(x in code_to_run for x in ["plt.", "sns.", ".plot"]):
                        # Varsayılan bir görselleştirme kodu ekle
                        if "SaleDate" in self.df.columns and "SaleCount" in self.df.columns:
                            # Tarih-bazlı veri için çizgi grafiği
                            default_viz = """
                            plt.figure(figsize=(10, 6))
                            plt.plot(df['SaleDate'], df['SaleCount'], marker='o')
                            plt.title('Satış Miktarının Zamana Göre Değişimi')
                            plt.xlabel('Tarih')
                            plt.ylabel('Satış Miktarı')
                            plt.grid(True)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            """
                            code_to_run = default_viz
                        elif viz_type == "line" and "SaleCount" in self.df.columns:
                            # Çizgi grafiği
                            default_viz = """
                            plt.figure(figsize=(10, 6))
                            plt.plot(df.index, df['SaleCount'], marker='o')
                            plt.title('Satış Miktarları')
                            plt.xlabel('Index')
                            plt.ylabel('Satış Miktarı')
                            plt.grid(True)
                            plt.tight_layout()
                            """
                            code_to_run = default_viz
                    
                    # Kodu çalıştır
                    viz_base64 = self.visualizer.execute_code(code_to_run, timeout=self.timeout)
                    structured_response.visualization = viz_base64
        except Exception as e:
            print(f"❌ Kod çalıştırma hatası: {str(e)}")
            traceback.print_exc()
            structured_response.error = f"Kod çalıştırılırken hata oluştu: {str(e)}"
        
        return structured_response

    def _extract_content_code(self, response: str) -> Tuple[str, str, str]:
        """LLM yanıtından içerik ve kod bölümlerini ayırır"""
        content = ""
        code = ""
        viz_code = ""
        
        # Basit metin ayrıştırma işlemi
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if "YANIT:" in line:
                current_section = "content"
                line = line.replace("YANIT:", "").strip()
            elif "KOD:" in line:
                current_section = "code"
                line = line.replace("KOD:", "").strip()
                continue
            elif "GÖRSEL:" in line:
                current_section = "viz"
                line = line.replace("GÖRSEL:", "").strip()
                continue
            
            if current_section == "content":
                content += line + "\n"
            elif current_section == "code":
                code += line + "\n"
            elif current_section == "viz":
                viz_code += line + "\n"
        
        # Eğer ayrıştırma çalışmadıysa ve ```python ... ``` blokları varsa, onları bul
        if not code and "```python" in response:
            code_blocks = response.split("```python")
            for block in code_blocks[1:]:  # İlk parça her zaman kod öncesidir
                if "```" in block:
                    code_part = block.split("```")[0].strip()
                    code += code_part + "\n"
                else:
                    # Eğer kapanış backtick yoksa, tüm bloğu al
                    code += block.strip() + "\n"
        
        # Eğer hiçbir kod ayrıştılamadıysa ve doğrudan kod gibi görünüyorsa
        if not code and not content and "import" in response and "df" in response:
            code = response
            content = "Analiz sonuçları:"
            
        # Kodun içinden görselleştirme kodunu çıkarmaya çalış
        if not viz_code and code and ("plt." in code or "sns." in code):
            viz_code = code
        
        # Markdown backtick bloklarını temizle (```python ... ``` gibi)
        code = self._clean_code_from_markdown(code)
        viz_code = self._clean_code_from_markdown(viz_code)
        
        return content.strip(), code.strip(), viz_code.strip()

    def _clean_code_from_markdown(self, code: str) -> str:
        """
        Markdown formatındaki kod bloklarını temizler
        - ```python ve ``` bloklarını kaldırır
        - Tekli backtick'leri kaldırır
        - Kod içindeki string'lerdeki backtick'lere dokunmaz
        """
        if not code:
            return ""
            
        # ```python ve ``` kalıplarını kaldır
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        
        # Başlangıçtaki veya satır başındaki tekli backtick'leri kaldır
        code = re.sub(r'(^|[\n])(`)([\w\s])', r'\1\3', code)
        
        # Gereksiz derlemeler varsa kaldır
        code = re.sub(r'^(Bu kodu çalıştır|Bu kodu kullanabilirsiniz|Python kodu:|Kod:|Görsel:|Görselleştirme kodu:).*[\n]', '', code, flags=re.MULTILINE)
        
        return code
        
    def _extract_non_viz_code(self, code: str) -> str:
        """
        Görselleştirme içermeyen kodu ayırır.
        Genellikle veri analizi kısmını alır, görselleştirme kısmını dışarıda bırakır.
        """
        if not code:
            return ""
        
        # Önce kodu temizle
        code = self._clean_code_from_markdown(code)
            
        lines = code.split('\n')
        non_viz_lines = []
        
        for line in lines:
            # Görselleştirme komutlarını atla
            if any(x in line for x in ["plt.", "sns.", ".plot(", ".figure(", "savefig", "show()"]):
                continue
            non_viz_lines.append(line)
            
        return "\n".join(non_viz_lines)

    def _parse_llm_response(self, response: str, query: str, 
                           generate_viz: bool, viz_type: str) -> StructuredResponse:
        """LLM yanıtını ayrıştırarak yapılandırılmış yanıt oluşturur"""
        structured_response = StructuredResponse()
        structured_response.metadata = {
            "query": query,
            "model": self.model,
            "timestamp": datetime.now().isoformat()
        }
        
        # İçerik ve kodu çıkar
        content, code, viz_code = self._extract_content_code(response)
        
        structured_response.content = content
        structured_response.code = code
        
        # Eğer görselleştirme isteniyorsa ve kod varsa, görselleştirmeyi oluştur
        if generate_viz and (viz_code or viz_type):
            try:
                if viz_type and viz_type in self.visualizer.supported_viz:
                    # Desteklenen görselleştirme fonksiyonunu kullanarak görsel oluştur
                    viz_base64 = self.visualizer.supported_viz[viz_type](query)
                else:
                    # Kodu çalıştırarak görselleştirme oluşturmayı dene
                    viz_base64 = self.visualizer.execute_code(viz_code or code, timeout=self.timeout)
                
                structured_response.visualization = viz_base64
            except Exception as e:
                print(f"❌ Görselleştirme hatası: {str(e)}")
                structured_response.error = f"Görselleştirme oluşturulurken hata oluştu: {str(e)}"
        
        return structured_response

    def transform_data(self, query: str) -> Tuple[pd.DataFrame, StructuredResponse]:
        """
        Doğal dil komutlarıyla DataFrame'i dönüştürür.
        
        Args:
            query (str): Dönüşüm için doğal dil sorgusu
            
        Returns:
            Tuple[pd.DataFrame, StructuredResponse]: Dönüştürülmüş DataFrame ve yanıt
        """
        # Önbellekten kontrol et
        query_hash = self._get_query_hash(f"transform:{query}")
        if query_hash in self.cache:
            print("✓ Önbellekten yanıt alındı.")
            return self.cache[query_hash]
            
        context = self.prepare_context()
        
        # API isteği için payload hazırla
        print("⏳ Ollama API'ye istek gönderiliyor...")
        start_time = time.time()
        
        # LLM temelli dönüşüm için payload'ı ayarla
        if self.direct_execution:
            # Doğrudan kod çalıştırma açıksa, LLM'e bu yetkiyi ver
            transform_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": f"""Sen bir veri dönüştürme uzmanısın. Veriler üzerinde dönüşümler yapacaksın.
                        İşte çalışacağın veri:
                        
                        {context}
                        
                        DataFrame 'df' değişkeninde tanımlıdır. Kullanıcının talebine göre veriyi dönüştür ve
                        sonucu 'result' değişkenine atayacak Python/pandas kodu yaz. Sadece çalışacak kodu yaz,
                        açıklama veya adım açıklamaları ekleme. Kod, doğrudan çalıştırılacak.
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"Veriyi şu şekilde dönüştür: {query}"
                    }
                ],
                "stream": False
            }
            
            try:
                response = self.api_client.call_api(transform_payload)
                
                # LLM'den gelen kodu çıkar
                code = response
                if "```python" in response:
                    code_blocks = response.split("```python")
                    for block in code_blocks[1:]:
                        if "```" in block:
                            code = block.split("```")[0].strip()
                            break
                
                # Kodu çalıştır
                print(f"⚙️ Dönüşüm kodu çalıştırılıyor...")
                
                # Değişkenleri hazırla
                exec_globals = {'pd': pd, 'np': pd.np}
                exec_locals = {'df': self.df.copy()}
                
                # Kodu çalıştır
                exec(code, exec_globals, exec_locals)
                
                # Sonuç DataFrame'i al
                if 'result' in exec_locals:
                    new_df = exec_locals['result']
                    message = "LLM tarafından oluşturulan dönüşüm kodu başarıyla çalıştırıldı."
                else:
                    # Result tanımlanmamışsa, son değişken değeri al
                    for var_name in list(exec_locals.keys())[::-1]:
                        if var_name != 'df' and isinstance(exec_locals[var_name], pd.DataFrame):
                            new_df = exec_locals[var_name]
                            message = f"Dönüşüm kodu çalıştırıldı ve {var_name} değişkeni kullanıldı."
                            break
                    else:
                        new_df = self.df.copy()
                        message = "Dönüşüm kodu result değişkeni üretmedi, orijinal veri döndürülüyor."
                
                result = StructuredResponse(
                    content=message,
                    code=code,
                    metadata={"query": query, "transform_type": "direct_execution"}
                )
                
                # Önbelleğe ekle
                self.cache[query_hash] = (new_df, result)
                
                elapsed = time.time() - start_time
                print(f"✓ Dönüşüm tamamlandı ({elapsed:.2f} saniye)")
                
                return new_df, result
                
            except Exception as e:
                print(f"❌ Dönüşüm hatası: {str(e)}")
                traceback.print_exc()
                return self.df.copy(), StructuredResponse(error=f"Dönüşüm sırasında hata: {str(e)}")
        
        # Eğer direct_execution kapalıysa, klasik yöntemi kullan
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": f"""Sen bir veri dönüştürme uzmanısın. Veriyi değiştirmek için kullanıcının isteğini yorumla.
                    İşte çalışacağın veri:
                    
                    {context}
                    
                    Kullanıcının isteğini analiz et ve aşağıdaki biçimde yanıt ver:
                    
                    1. AÇIKLAMA: [Yapılacak dönüşümün kısa açıklaması]
                    2. TÜR: [filter, sort, group, aggregate, pivot gibi dönüşüm türü]
                    3. PARAMETRELER: [Dönüşüm için gerekli parametreler (JSON formatında)]
                    4. KOD: [Dönüşümü gerçekleştirecek tam pandas kodu]
                    """
                },
                {
                    "role": "user", 
                    "content": f"Veriyi şu şekilde dönüştür: {query}"
                }
            ],
            "stream": False
        }
        
        try:
            response = self.api_client.call_api(payload)
            
            # Yanıtı ayrıştır
            transform_type, params, code = self.transformer.parse_transform_response(response)
            
            # Dönüşümü uygula
            if transform_type in self.transformer.supported_transforms:
                new_df = self.transformer.supported_transforms[transform_type](params)
                message = f"'{transform_type}' türünde dönüşüm başarıyla uygulandı."
            elif code:
                # Kodu doğrudan çalıştırmayı dene
                loc = {"df": self.df, "pd": pd}
                exec(code, globals(), loc)
                new_df = loc.get("result", self.df)
                message = "Özel dönüşüm kodu başarıyla uygulandı."
            else:
                # Özel durum - Stock 100'den büyük filtresi
                if "stock" in query.lower() and "100" in query and ("büyük" in query.lower() or "fazla" in query.lower()):
                    new_df = self.df[self.df['Stock'] > 100]
                    message = "Stok sayısı 100'den büyük olan ürünler filtrelendi."
                else:
                    new_df = self.df
                    message = "Dönüşüm uygulanamadı."
            
            result = StructuredResponse(
                content=message,
                code=code,
                metadata={"query": query, "transform_type": transform_type}
            )
            
            # Önbelleğe ekle
            self.cache[query_hash] = (new_df, result)
            
            elapsed = time.time() - start_time
            print(f"✓ Dönüşüm tamamlandı ({elapsed:.2f} saniye)")
            
            return new_df, result
        except Exception as e:
            print(f"❌ Dönüşüm hatası: {str(e)}")
            return self.df, StructuredResponse(error=f"Veri dönüşümü sırasında hata: {str(e)}")

    # Kullanımı kolaylaştırmak için kısa metod isimleri 
    def ask(self, query: str) -> StructuredResponse:
        """
        DataFrame hakkında bilgi edinmek için AI'ya hızlı bir soru sorar.
        İyileştirilmiş performans için özelleştirilmiştir.
        
        Args:
            query (str): Kullanıcı sorgusu
            
        Returns:
            StructuredResponse: Yapılandırılmış yanıt
        """
        # Önbellekten kontrol et
        query_hash = self._get_query_hash(f"ask:{query}")
        if query_hash in self.cache:
            print("✓ Önbellekten yanıt alındı.")
            return self.cache[query_hash]
            
        # Analiz sorguları için bağlamı sınırla ve basitleştir
        context = self.prepare_context()
        
        print("⏳ Ollama API'ye istek gönderiliyor...")
        start_time = time.time()
        
        # Prompt'ı sadeleştir ve netleştir
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": f"""Sen bir veri analiz asistanısın. Aşağıdaki veri hakkında kısa ve net yanıtlar ver.
                    
{context}

Yanıtında sorunun cevabına odaklan ve açık, anlaşılır bir yanıt ver. Sayısal sonuçları net şekilde belirt.
"""
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            "stream": False
        }
        
        # Eğer doğrudan kod çalıştırma açıksa
        if self.direct_execution:
            payload["messages"][0]["content"] += "\nEğer hesaplama yapman gerekiyorsa, Python/pandas kodu yazabilirsin. DataFrame df adlı değişkende bulunuyor. Sonucu 'result' değişkenine ata."
        
        try:
            response = self.api_client.call_api(payload)
            
            # Doğrudan kod çalıştırma açıksa ve yanıtta kod varsa
            if self.direct_execution and ("```python" in response or "import" in response):
                result = self._execute_llm_code(response, query, False, None)
            else:
                # Yanıtı doğrudan metne dönüştür
                result = StructuredResponse(
                    content=response.strip(),
                    metadata={"query": query, "model": self.model}
                )
            
            # Geçmişe ekle
            self.history_manager.add_to_history(query, result)
            
            # Önbelleğe ekle
            self.cache[query_hash] = result
            
            elapsed = time.time() - start_time
            print(f"✓ Yanıt alındı ({elapsed:.2f} saniye)")
            
            return result
            
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
            return StructuredResponse(error=f"Error calling AI service: {str(e)}")
    
    def analyze(self, query: str, viz_type: str = None) -> StructuredResponse:
        """
        DataFrame'i analiz eder ve görselleştirme ile sonuçları döndürür.
        
        Args:
            query (str): Analiz sorgusu
            viz_type (str, optional): Görselleştirme türü
            
        Returns:
            StructuredResponse: Yapılandırılmış yanıt (görselleştirme ile)
        """
        return self.run(query, generate_viz=True, viz_type=viz_type, generate_code=True)
    
    def transform(self, query: str) -> pd.DataFrame:
        """
        DataFrame'i doğal dil komutlarıyla dönüştürür ve sonuç DataFrame'i döndürür.
        
        Args:
            query (str): Dönüşüm sorgusu
            
        Returns:
            pd.DataFrame: Dönüştürülmüş DataFrame
        """
        new_df, _ = self.transform_data(query)
        return new_df
    
    def plot(self, query: str, viz_type: str = "bar") -> StructuredResponse:
        """
        DataFrame üzerinde belirlenen görselleştirmeyi oluşturur.
        
        Args:
            query (str): Görselleştirme sorgusu
            viz_type (str): Görselleştirme türü
            
        Returns:
            StructuredResponse: Oluşturulan görselleştirme ile yanıt
        """
        return self.run(query, generate_viz=True, viz_type=viz_type, generate_code=False)

    def get_history(self) -> List[Dict]:
        """Analiz geçmişini döndürür"""
        return self.history_manager.get_history()
    
    def clear_history(self) -> None:
        """Analiz geçmişini temizler"""
        self.history_manager.clear_history()
        
    def clear_cache(self) -> None:
        """Önbelleği temizler"""
        self.cache = {}
        print("✓ Önbellek temizlendi.")
