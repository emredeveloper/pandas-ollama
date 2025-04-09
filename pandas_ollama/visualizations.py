"""
Veri görselleştirme için fonksiyonlar içeren modül
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import traceback
import signal
import warnings
import re
from typing import Optional, Dict, List, Callable, Any
from contextlib import contextmanager
from datetime import datetime

class TimeoutException(Exception):
    """Zaman aşımı hatası için özel istisna sınıfı"""
    pass

@contextmanager
def time_limit(seconds):
    """İşlemi belirli bir süre sonra sonlandırır"""
    def signal_handler(signum, frame):
        raise TimeoutException("Zaman aşımı (timeout)!")
        
    # Signal sadece Unix tabanlı sistemlerde çalışır (Linux, macOS)
    # Windows'ta çalışmaz, bu nedenle try-except bloğu içinde çağırıyoruz
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        yield
    except (ValueError, AttributeError):
        # Windows veya Signal kullanılamayan sistemlerde sessizce geç
        yield
    finally:
        try:
            signal.alarm(0)
        except (ValueError, AttributeError):
            pass

class Visualizer:
    """Pandas DataFrame görselleştirme sınıfı"""
    
    def __init__(self, dataframe: pd.DataFrame, timeout: int = 30):
        self.df = dataframe
        self.timeout = timeout
        
        # Sütun türlerini belirle
        self._analyze_datatypes()
        
        # Desteklenen grafikler
        self.supported_viz = {
            "bar": self.plot_bar,
            "line": self.plot_line,
            "scatter": self.plot_scatter,
            "hist": self.plot_histogram,
            "pie": self.plot_pie,
            "heatmap": self.plot_heatmap,
            "box": self.plot_boxplot,
            "count": self.plot_countplot,   # Yeni: Kategorik sayım grafiği
            "area": self.plot_area,         # Yeni: Alan grafiği
            "density": self.plot_density,   # Yeni: Yoğunluk grafiği
            "violin": self.plot_violin      # Yeni: Keman grafiği
        }
    
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
    
    def _analyze_datatypes(self) -> None:
        """DataFrame'deki veri tipleri hakkında bilgi toplar"""
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Tarih sütunlarını tespit et
        self.datetime_columns = []
        
        # Önce zaten datetime tipinde olan sütunları ekle
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.datetime_columns.append(col)
        
        # Ardından string olup tarih formatında olabilecek sütunları kontrol et
        for col in self.categorical_columns:
            if col not in self.datetime_columns and (self._is_likely_date_column(col) or 
               any(isinstance(s, str) and ('/' in s or '-' in s) for s in self.df[col].head().values)):
                try:
                    # Tarih formatını tespit etmeye çalış
                    sample_values = self.df[col].dropna().head(10).values
                    date_format = self._detect_date_format(sample_values)
                    
                    # Eğer format tespit edildiyse, o formatı kullan
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if date_format:
                            pd.to_datetime(self.df[col], format=date_format, errors='raise')
                        else:
                            pd.to_datetime(self.df[col], errors='raise')
                            
                    self.datetime_columns.append(col)
                except:
                    pass
        
        # Boolean sütunları tespit et
        self.bool_columns = self.df.select_dtypes(include=['bool']).columns.tolist()
    
    def _get_best_columns_for_visualization(self, viz_type: str, query: str = "") -> Dict:
        """Görselleştirme için en uygun sütunları seçer"""
        params = {}
        
        if viz_type == "scatter":
            # Scatter plot için iki sayısal sütun gerekli
            if len(self.numeric_columns) >= 2:
                params["x"] = self.numeric_columns[0]
                params["y"] = self.numeric_columns[1]
        elif viz_type == "hist":
            # Histogram için bir sayısal sütun gerekli
            if self.numeric_columns:
                params["x"] = self.numeric_columns[0]
        elif viz_type == "pie":
            # Pasta grafiği için kategorik sütun gerekli
            if self.categorical_columns:
                params["names"] = self.categorical_columns[0]
        elif viz_type == "heatmap":
            # Isı haritası için korelasyon
            if len(self.numeric_columns) > 1:
                params["data"] = "correlation"
                params["columns"] = self.numeric_columns
        elif viz_type == "box":
            # Kutu grafiği için kategorik ve sayısal sütun gerekli
            if self.categorical_columns and self.numeric_columns:
                params["x"] = self.categorical_columns[0]
                params["y"] = self.numeric_columns[0]
        
        return params
    
    def _clean_code(self, code: str) -> str:
        """
        Markdown formatındaki kod bloklarını temizler
        - ```python ve ``` bloklarını kaldırır
        - Tekli backtick'leri kaldırır
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
    
    def execute_code(self, code: str, timeout: int = None) -> Optional[str]:
        """
        Verilen görselleştirme kodunu güvenli bir şekilde çalıştırır ve base64 kodlu görüntüyü döndürür.
        
        Args:
            code (str): Çalıştırılacak Python kodu
            timeout (int, optional): İşlem zaman aşımı süresi (saniye)
            
        Returns:
            Optional[str]: Base64 kodlu görüntü verisi
        """
        if not code:
            return None
        
        # Markdown formatını temizle
        code = self._clean_code(code)
        
        # Timeout belirlenmemişse instance değerini kullan
        if timeout is None:
            timeout = self.timeout
            
        try:
            # Kodu temizle ve güvenlik kontrolü yap
            if "import " in code and not any(x in code for x in ["import matplotlib", "import seaborn", "import pandas", "import numpy"]):
                # Güvenlik için sadece belirli importlara izin ver
                raise ValueError("Güvenlik nedeniyle sadece matplotlib, pandas, numpy ve seaborn kütüphaneleri kullanılabilir.")
            
            # Mevcut DataFrame'i kullanacak şekilde kodu düzenle
            modified_code = code.replace("df = pd.DataFrame", "# df zaten tanımlı")
            
            # Görüntü buffer'ı oluştur
            buffer = io.BytesIO()
            
            # Locals ve globals sözlüklerini hazırla
            import numpy as np
            loc = {
                "plt": plt, 
                "sns": sns, 
                "pd": pd, 
                "np": np,
                "df": self.df, 
                "buffer": buffer
            }
            
            # Zaman aşımı ile kodu çalıştır
            with time_limit(timeout):
                # Kodu çalıştır (varsayılan olarak matplotlib figürü oluşturacak)
                print(f"⚙️ Görselleştirme kodu çalıştırılıyor (zaman aşımı: {timeout} saniye)...")
                
                # Eğer kodda fig, figure veya plt.figure() kullanılmamışsa
                # varsayılan bir figure oluştur
                if not re.search(r'(plt\.figure|fig\s*=|figure\s*=)', modified_code):
                    plt.figure(figsize=(10, 6))
                    
                exec(modified_code, globals(), loc)
                
                # Figürün kaydedildiğinden emin ol
                try:
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    plt.close('all')  # Tüm figürleri kapat
                except Exception as e:
                    print(f"⚠️ Figure kayıt hatası: {str(e)}")
                    # Belki kod zaten kaydetmiştir veya figür oluşturmamıştır
                    pass
                
                # Base64 kodlaması yap
                buffer.seek(0)
                image_data = buffer.read()
                
                # Eğer buffer boşsa veya geçersizse hata ver
                if not image_data or len(image_data) < 100:  # Çok küçük dosyalar muhtemelen geçersizdir
                    plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, "Görselleştirme oluşturulamadı", 
                             horizontalalignment='center', verticalalignment='center', fontsize=14)
                    plt.axis('off')
                    plt.savefig(buffer, format='png')
                    plt.close()
                    buffer.seek(0)
                    image_data = buffer.read()
                
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                return image_base64
                
        except TimeoutException:
            print(f"⚠️ Görselleştirme zaman aşımına uğradı (> {timeout} saniye)")
            # Zaman aşımı için hata görseli oluştur
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Görselleştirme zaman aşımına uğradı (> {timeout} saniye)", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
            
        except Exception as e:
            print(f"❌ Görselleştirme hatası: {str(e)}")
            traceback.print_exc()
            
            # Hata görseli oluştur
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Hata: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

    def plot_bar(self, query: str) -> Optional[str]:
        """Bar grafik oluşturur"""
        try:
            if not self.numeric_columns or not self.categorical_columns:
                return self._create_error_viz("Uygun sütunlar bulunamadı")
                
            x = self.categorical_columns[0]
            y = self.numeric_columns[0]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=x, y=y, data=self.df)
            plt.title(f"{x} - {y} Bar Grafiği")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Bar grafik hatası: {str(e)}")
            return self._create_error_viz(f"Bar grafik hatası: {str(e)}")

    def _create_error_viz(self, error_message: str) -> str:
        """Hata mesajı içeren görselleştirme oluşturur"""
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, error_message, 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    # Yeni görselleştirme türleri
    def plot_countplot(self, query: str) -> Optional[str]:
        """Kategorik sütunların sayımını gösteren grafik"""
        try:
            if not self.categorical_columns:
                return self._create_error_viz("Kategorik sütun bulunamadı")
                
            cat_col = self.categorical_columns[0]
            
            plt.figure(figsize=(10, 6))
            sns.countplot(x=cat_col, data=self.df)
            plt.title(f"{cat_col} Değer Sayıları")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Count plot hatası: {str(e)}")
            return self._create_error_viz(f"Count plot hatası: {str(e)}")

    def plot_area(self, query: str) -> Optional[str]:
        """Alan grafiği oluşturur"""
        try:
            if not self.numeric_columns or len(self.numeric_columns) < 2:
                return self._create_error_viz("Yeterli sayısal sütun bulunamadı")
                
            plt.figure(figsize=(12, 6))
            self.df[self.numeric_columns].plot.area(alpha=0.5)
            plt.title("Sayısal Değişkenler Alan Grafiği")
            plt.grid(True)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Alan grafik hatası: {str(e)}")
            return self._create_error_viz(f"Alan grafik hatası: {str(e)}")

    def plot_density(self, query: str) -> Optional[str]:
        """Yoğunluk grafiği (KDE) oluşturur"""
        try:
            if not self.numeric_columns:
                return self._create_error_viz("Sayısal sütun bulunamadı")
                
            plt.figure(figsize=(10, 6))
            for col in self.numeric_columns[:3]:  # En fazla 3 sütun göster
                sns.kdeplot(self.df[col], label=col)
            plt.title("Sayısal Değişkenler Yoğunluk Grafiği")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Yoğunluk grafik hatası: {str(e)}")
            return self._create_error_viz(f"Yoğunluk grafik hatası: {str(e)}")

    def plot_violin(self, query: str) -> Optional[str]:
        """Keman grafiği oluşturur"""
        try:
            if not self.numeric_columns or not self.categorical_columns:
                return self._create_error_viz("Uygun sütunlar bulunamadı")
                
            x = self.categorical_columns[0]
            y = self.numeric_columns[0]
            
            plt.figure(figsize=(12, 6))
            sns.violinplot(x=x, y=y, data=self.df)
            plt.title(f"{x} - {y} Keman Grafiği")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Keman grafik hatası: {str(e)}")
            return self._create_error_viz(f"Keman grafik hatası: {str(e)}")

    def plot_line(self, query: str) -> Optional[str]:
        """Çizgi grafik oluşturur"""
        try:
            # Zamansal veri için uygun mu kontrol et
            if self.datetime_columns:
                x = self.datetime_columns[0]
                
                # İlk sayısal sütunu bul
                y = None
                for col in self.numeric_columns:
                    y = col
                    break
                
                if not y:
                    return self._create_error_viz("Sayısal sütun bulunamadı")
                
                plt.figure(figsize=(12, 6))
                plt.plot(self.df[x], self.df[y], marker='o', linestyle='-')
                plt.title(f"{y} Zaman Serisi")
                plt.grid(True)
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
            
            # Zamansal veri yoksa, indeksi veya sıralı sayısal değer kullan
            elif self.numeric_columns:
                # En uygun sütunları belirle
                y = self.numeric_columns[0]
                
                plt.figure(figsize=(12, 6))
                plt.plot(self.df.index, self.df[y], marker='o', linestyle='-')
                plt.title(f"{y} Seri Grafiği")
                plt.grid(True)
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
                
            return self._create_error_viz("Görselleştirme için uygun sütun bulunamadı")
        except Exception as e:
            print(f"Çizgi grafik hatası: {str(e)}")
            return self._create_error_viz(f"Çizgi grafik hatası: {str(e)}")

    def plot_scatter(self, query: str) -> Optional[str]:
        """Dağılım grafiği oluşturur"""
        try:
            # En uygun sütunları belirle
            params = self._get_best_columns_for_visualization("scatter", query)
            
            if not params or "x" not in params or "y" not in params:
                return self._create_error_viz("Sayısal sütunlar bulunamadı")
                
            x = params["x"]
            y = params["y"]
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=x, y=y, data=self.df)
            plt.title(f"{x} - {y} Dağılım Grafiği")
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Dağılım grafiği hatası: {str(e)}")
            return self._create_error_viz(f"Dağılım grafiği hatası: {str(e)}")

    def plot_histogram(self, query: str) -> Optional[str]:
        """Histogram oluşturur"""
        try:
            # En uygun sütunları belirle
            params = self._get_best_columns_for_visualization("hist", query)
            
            if not params or "x" not in params:
                return self._create_error_viz("Sayısal sütun bulunamadı")
                
            x = params["x"]
            
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[x], kde=True)
            plt.title(f"{x} Dağılımı")
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Histogram hatası: {str(e)}")
            return self._create_error_viz(f"Histogram hatası: {str(e)}")

    def plot_pie(self, query: str) -> Optional[str]:
        """Pasta grafiği oluşturur"""
        try:
            # En uygun sütunları belirle
            params = self._get_best_columns_for_visualization("pie", query)
            
            if not params or "names" not in params:
                return self._create_error_viz("Kategorik sütun bulunamadı")
                
            names = params["names"]
            
            counts = self.df[names].value_counts()
            
            plt.figure(figsize=(10, 8))
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
            plt.title(f"{names} Dağılımı")
            plt.axis('equal')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Pasta grafiği hatası: {str(e)}")
            return self._create_error_viz(f"Pasta grafiği hatası: {str(e)}")

    def plot_heatmap(self, query: str) -> Optional[str]:
        """Isı haritası oluşturur"""
        try:
            # En uygun sütunları belirle
            params = self._get_best_columns_for_visualization("heatmap", query)
            
            if not params or "data" not in params or params["data"] != "correlation":
                return self._create_error_viz("Korelasyon için yeterli sayısal sütun bulunamadı")
                
            columns = params["columns"]
            
            plt.figure(figsize=(12, 10))
            correlation = self.df[columns].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Korelasyon Isı Haritası")
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Isı haritası hatası: {str(e)}")
            return self._create_error_viz(f"Isı haritası hatası: {str(e)}")

    def plot_boxplot(self, query: str) -> Optional[str]:
        """Kutu grafiği oluşturur"""
        try:
            # En uygun sütunları belirle
            params = self._get_best_columns_for_visualization("box", query)
            
            if not params or "x" not in params or "y" not in params:
                return self._create_error_viz("Uygun sütunlar bulunamadı")
                
            x = params["x"]
            y = params["y"]
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=x, y=y, data=self.df)
            plt.title(f"{x} - {y} Kutu Grafiği")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Kutu grafiği hatası: {str(e)}")
            return self._create_error_viz(f"Kutu grafiği hatası: {str(e)}")
