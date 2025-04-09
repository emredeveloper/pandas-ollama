"""
Veri dönüşüm işlemleri için modül - CSV ve genel verilere uyumlu
"""

import pandas as pd
import re
import json
from typing import Dict, Tuple, List, Any, Optional

class DataTransformer:
    """Pandas DataFrame dönüşüm sınıfı - Herhangi bir veri yapısı için uyumlu"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
        # Desteklenen veri dönüşümleri
        self.supported_transforms = {
            "filter": self.transform_filter,
            "sort": self.transform_sort,
            "group": self.transform_group,
            "aggregate": self.transform_aggregate,
            "pivot": self.transform_pivot,
            "select": self.transform_select,  # Yeni: Belirli sütunları seçme
            "rename": self.transform_rename,  # Yeni: Sütunları yeniden adlandırma
            "fillna": self.transform_fillna,  # Yeni: Eksik değerleri doldurma
            "dropna": self.transform_dropna   # Yeni: Eksik değerleri silme
        }
        
        # Veri tipleri hakkında bilgi
        self._analyze_datatypes()
    
    def _analyze_datatypes(self) -> None:
        """DataFrame'deki veri tipleri hakkında bilgi toplar"""
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = self.df.select_dtypes(include=['datetime']).columns.tolist()
        self.bool_columns = self.df.select_dtypes(include=['bool']).columns.tolist()
        
        # Potansiyel anahtar sütunları tespit et (benzersiz değerlere sahip)
        self.key_columns = []
        for col in self.df.columns:
            if self.df[col].nunique() == len(self.df) and self.df[col].nunique() > 1:
                self.key_columns.append(col)
    
    def parse_transform_response(self, response: str) -> Tuple[str, Dict, str]:
        """Dönüşüm yanıtını ayrıştırır"""
        transform_type = ""
        params = {}
        code = ""
        
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if "TÜR:" in line:
                current_section = "type"
                transform_type = line.replace("TÜR:", "").strip().lower()
            elif "PARAMETRELER:" in line:
                current_section = "params"
                param_text = line.replace("PARAMETRELER:", "").strip()
                try:
                    # JSON formatında parametreleri ayrıştırmayı dene
                    if param_text.startswith("{") and param_text.endswith("}"):
                        import json
                        params = json.loads(param_text)
                except:
                    pass
            elif "KOD:" in line:
                current_section = "code"
                continue
            
            if current_section == "code":
                code += line + "\n"
        
        # Genel filtreleme durumları için akıllı tespit
        if not transform_type or not params:
            transform_type, params = self._detect_transformation_from_text(response)
        
        return transform_type, params, code.strip()
    
    def _detect_transformation_from_text(self, text: str) -> Tuple[str, Dict]:
        """Metinden dönüşüm türünü ve parametreleri tespit eder"""
        text = text.lower()
        
        # Filtreleme işlemi tespiti
        if any(x in text for x in ["filtre", "filter", "where", "seç", "bul", "getir", "göster", "seçin"]):
            column, condition, value = self._extract_filter_params(text)
            if column:
                return "filter", {"column": column, "condition": condition, "value": value}
        
        # Sıralama işlemi tespiti
        if any(x in text for x in ["sırala", "sort", "order", "düzenle"]):
            column, ascending = self._extract_sort_params(text)
            if column:
                return "sort", {"column": column, "ascending": ascending}
        
        # Gruplama işlemi tespiti
        if any(x in text for x in ["grup", "group", "topla", "aggregate"]):
            columns, agg_func = self._extract_group_params(text)
            if columns:
                return "group", {"columns": columns, "agg_func": agg_func}
        
        # Sütun seçme işlemi tespiti
        if any(x in text for x in ["sütun seç", "select column", "columns", "sütunlar"]):
            columns = self._extract_columns_from_text(text)
            if columns:
                return "select", {"columns": columns}
        
        # Eksik değer işlemi tespiti
        if any(x in text for x in ["eksik", "missing", "null", "na", "boş"]):
            if any(x in text for x in ["doldur", "fill", "değiştir", "replace"]):
                return "fillna", {"value": 0}  # Varsayılan olarak 0 ile doldur
            elif any(x in text for x in ["kaldır", "drop", "sil", "çıkar"]):
                return "dropna", {"how": "any"}
                
        return "", {}
    
    def _extract_filter_params(self, text: str) -> Tuple[str, str, Any]:
        """Metin içinden filtreleme parametrelerini çıkarır"""
        # Tüm sütunlar için kontrol et
        column = None
        best_match_pos = float('inf')
        
        for col in self.df.columns:
            col_lower = col.lower()
            if col_lower in text:
                # Eğer birden fazla sütun eşleşirse, metinde daha erken geçeni seç
                col_pos = text.find(col_lower)
                if col_pos < best_match_pos:
                    column = col
                    best_match_pos = col_pos
        
        # Eğer sütun bulunamadıysa, en uygun sütunu tahmin et
        if not column:
            # Metinde geçen kelimelerle sütun isimlerini karşılaştır
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                if len(word) > 3:  # En az 4 karakter olan kelimeleri kontrol et
                    for col in self.df.columns:
                        if word.lower() in col.lower():
                            column = col
                            break
                    if column:
                        break
        
        # Yine bulunamadıysa ve sayısal filtre gibi görünüyorsa, ilk sayısal sütunu kullan
        if not column and re.search(r'\d+', text) and self.numeric_columns:
            column = self.numeric_columns[0]
        
        # Koşul operatörünü bul
        condition = "=="  # Varsayılan eşitlik
        
        # Karşılaştırma operatörlerini kontrol et
        if any(x in text for x in ["büyük", "fazla", "yukarı", "üstünde", ">"]):
            condition = ">"
        elif any(x in text for x in ["küçük", "az", "aşağı", "altında", "<"]):
            condition = "<"
        elif any(x in text for x in ["eşit değil", "dışında", "hariç", "!="]):
            condition = "!="
        elif any(x in text for x in ["büyük eşit", "en az", "minimum", ">="]):
            condition = ">="
        elif any(x in text for x in ["küçük eşit", "en fazla", "maksimum", "<="]):
            condition = "<="
        elif any(x in text for x in ["içeren", "içerir", "var", "bulunur"]):
            condition = "contains"
        
        # Değeri bul
        value = None
        
        # Sayısal değeri bul
        if column and column in self.numeric_columns:
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if numbers:
                # En uygun sayıyı seç (koşul operatörüne yakın olanı)
                cond_terms = ["büyük", "küçük", "eşit", "fazla", "az"]
                best_num_pos = float('inf')
                best_num = None
                
                for num in numbers:
                    for term in cond_terms:
                        term_pos = text.find(term)
                        if term_pos > -1:
                            num_pos = text.find(num)
                            distance = abs(num_pos - term_pos)
                            if distance < best_num_pos:
                                best_num_pos = distance
                                best_num = num
                
                if best_num:
                    if '.' in best_num:
                        value = float(best_num)
                    else:
                        value = int(best_num)
                else:
                    value = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
        
        # Kategorik değeri bul
        elif column and column in self.categorical_columns:
            # Olası değerleri al
            possible_values = self.df[column].unique()
            
            # Metindeki değeri bul
            for val in possible_values:
                if str(val).lower() in text:
                    value = val
                    break
            
            # Bulunamadıysa, metinde sık geçen bir kelimeyi dene
            if value is None:
                words = re.findall(r'\b\w+\b', text)
                word_counts = {}
                for word in words:
                    if len(word) > 3:  # 3 karakterden uzun kelimeler
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                # En sık geçen kelimeyi bul
                if word_counts:
                    most_common = max(word_counts.items(), key=lambda x: x[1])[0]
                    value = most_common
        
        # Değer hala bulunamadıysa, varsayılan bir değer kullan
        if value is None and column:
            if column in self.numeric_columns:
                # Sayısal sütun için ortalama değer
                value = self.df[column].mean()
            elif column in self.categorical_columns:
                # Kategorik sütun için en sık değer
                value = self.df[column].mode()[0]
        
        return column, condition, value
    
    def _extract_sort_params(self, text: str) -> Tuple[str, bool]:
        """Metin içinden sıralama parametrelerini çıkarır"""
        column = None
        
        # Sütun adını bul - önce tam eşleşmeleri dene
        for col in self.df.columns:
            if col.lower() in text.lower():
                column = col
                break
        
        # Eğer bulunamadıysa, sayısal bir sütun bulmayı dene
        if not column and self.numeric_columns:
            # Metin "büyükten küçüğe" veya "azalan" içeriyorsa, muhtemelen sayısal bir sütun isteniyor
            if any(x in text.lower() for x in ["büyükten", "azalan", "desc"]):
                column = self.numeric_columns[0]
            else:
                column = self.numeric_columns[0]
        
        # Yine bulunamadıysa, herhangi bir sütunu kullan
        if not column and len(self.df.columns) > 0:
            column = self.df.columns[0]
        
        # Sıralama yönünü bul
        ascending = True  # Varsayılan artan sıralama
        if any(x in text.lower() for x in ["azalan", "büyükten küçüğe", "desc", "tersine", "ters", "tersten"]):
            ascending = False
        
        return column, ascending
    
    def _extract_group_params(self, text: str) -> Tuple[list, str]:
        """Metin içinden gruplama parametrelerini çıkarır"""
        columns = []
        
        # Kategorik sütunları öncelikle kontrol et
        for col in self.categorical_columns:
            if col.lower() in text.lower():
                columns.append(col)
        
        # Eğer kategorik sütun bulunamadıysa, herhangi bir sütunu dene
        if not columns:
            for col in self.df.columns:
                if col.lower() in text.lower():
                    columns.append(col)
                    break
        
        # Yine bulunamadıysa ve kategorik sütunlar varsa, ilkini kullan
        if not columns and self.categorical_columns:
            columns = [self.categorical_columns[0]]
        
        # Aggregasyon fonksiyonunu bul
        agg_func = "mean"  # Varsayılan ortalama
        
        if any(x in text.lower() for x in ["toplam", "sum", "topla"]):
            agg_func = "sum"
        elif any(x in text.lower() for x in ["sayı", "count", "adet", "miktar"]):
            agg_func = "count"
        elif any(x in text.lower() for x in ["max", "en büyük", "maksimum", "en yüksek"]):
            agg_func = "max"
        elif any(x in text.lower() for x in ["min", "en küçük", "minimum", "en düşük"]):
            agg_func = "min"
        elif any(x in text.lower() for x in ["ort", "ortalama", "mean", "avg", "average"]):
            agg_func = "mean"
        
        return columns, agg_func
    
    def _extract_columns_from_text(self, text: str) -> List[str]:
        """Metinden sütun adlarını çıkarır"""
        columns = []
        
        # Tüm sütunları kontrol et
        for col in self.df.columns:
            if col.lower() in text.lower():
                columns.append(col)
        
        # Eğer hiç sütun bulunamadıysa, metinde geçen kelimeleri kontrol et
        if not columns:
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                if len(word) > 3:  # En az 4 karakter olan kelimeleri kontrol et
                    for col in self.df.columns:
                        if word.lower() in col.lower() or col.lower() in word.lower():
                            columns.append(col)
        
        # Tekrar eden sütunları temizle
        columns = list(dict.fromkeys(columns))
        
        return columns
    
    def transform_filter(self, params: Dict) -> pd.DataFrame:
        """DataFrame'i filtreleme"""
        column = params.get("column")
        condition = params.get("condition", "==")
        value = params.get("value")
        
        if not column or value is None or column not in self.df.columns:
            return self.df
            
        if condition == "==":
            return self.df[self.df[column] == value]
        elif condition == "!=":
            return self.df[self.df[column] != value]
        elif condition == ">":
            return self.df[self.df[column] > value]
        elif condition == "<":
            return self.df[self.df[column] < value]
        elif condition == ">=":
            return self.df[self.df[column] >= value]
        elif condition == "<=":
            return self.df[self.df[column] <= value]
        elif condition == "in":
            if isinstance(value, list):
                return self.df[self.df[column].isin(value)]
        elif condition == "contains":
            # Hem string hem de listeler için çalışabilir
            if isinstance(value, str):
                # Eğer sütun string tipindeyse contains kullan
                if self.df[column].dtype == 'object':
                    return self.df[self.df[column].str.contains(str(value), na=False, case=False)]
                else:
                    # Değilse string'e çevirip kontrol et
                    return self.df[self.df[column].astype(str).str.contains(str(value), na=False, case=False)]
        
        return self.df

    def transform_sort(self, params: Dict) -> pd.DataFrame:
        """DataFrame'i sıralama"""
        column = params.get("column")
        ascending = params.get("ascending", True)
        
        if not column or column not in self.df.columns:
            return self.df
            
        return self.df.sort_values(by=column, ascending=ascending)

    def transform_group(self, params: Dict) -> pd.DataFrame:
        """DataFrame'i gruplama"""
        columns = params.get("columns")
        agg_func = params.get("agg_func", "mean")
        
        if not columns:
            return self.df
            
        if isinstance(columns, str):
            columns = [columns]
        
        # Sütunların varlığını kontrol et
        valid_columns = [col for col in columns if col in self.df.columns]
        if not valid_columns:
            return self.df
            
        # Hangi sütunların toplanacağını otomatik tespit et
        if agg_func in ["mean", "sum", "min", "max"]:
            # Sayısal sütunları kullan
            agg_columns = [col for col in self.numeric_columns if col not in valid_columns]
            if not agg_columns:
                # Sayısal sütun yoksa, count kullan
                return self.df.groupby(valid_columns).size().reset_index(name='count')
            
            # Agrege fonksiyonları için sözlük oluştur
            agg_dict = {col: agg_func for col in agg_columns}
            return self.df.groupby(valid_columns).agg(agg_dict).reset_index()
        else:
            # count gibi diğer agregasyon fonksiyonları için
            return self.df.groupby(valid_columns).agg(agg_func).reset_index()

    # Yeni dönüşüm fonksiyonları
    def transform_select(self, params: Dict) -> pd.DataFrame:
        """Belirli sütunları seçme"""
        columns = params.get("columns")
        
        if not columns:
            return self.df
            
        if isinstance(columns, str):
            columns = [columns]
        
        # Sütunların varlığını kontrol et
        valid_columns = [col for col in columns if col in self.df.columns]
        if not valid_columns:
            return self.df
            
        return self.df[valid_columns]
    
    def transform_rename(self, params: Dict) -> pd.DataFrame:
        """Sütunları yeniden adlandırma"""
        rename_dict = params.get("rename_dict", {})
        
        if not rename_dict:
            return self.df
            
        # Sütunların varlığını kontrol et
        valid_renames = {old: new for old, new in rename_dict.items() if old in self.df.columns}
        if not valid_renames:
            return self.df
            
        return self.df.rename(columns=valid_renames)
    
    def transform_fillna(self, params: Dict) -> pd.DataFrame:
        """Eksik değerleri doldurma"""
        value = params.get("value")
        columns = params.get("columns")
        
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            # Sütunların varlığını kontrol et
            valid_columns = [col for col in columns if col in self.df.columns]
            if not valid_columns:
                return self.df
                
            # Sadece belirtilen sütunlarda doldur
            result = self.df.copy()
            for col in valid_columns:
                result[col] = result[col].fillna(value)
            return result
        else:
            # Tüm DataFrame'de doldur
            return self.df.fillna(value)
    
    def transform_dropna(self, params: Dict) -> pd.DataFrame:
        """Eksik değerleri silme"""
        how = params.get("how", "any")  # 'any' veya 'all'
        columns = params.get("columns")
        
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            # Sütunların varlığını kontrol et
            valid_columns = [col for col in columns if col in self.df.columns]
            if not valid_columns:
                return self.df
                
            # Sadece belirtilen sütunlarda eksik değerleri kontrol et
            return self.df.dropna(subset=valid_columns, how=how)
        else:
            # Tüm DataFrame'deki eksik değerleri sil
            return self.df.dropna(how=how)
    
    # Mevcut dönüşüm fonksiyonları...
    def transform_aggregate(self, params: Dict) -> pd.DataFrame:
        """DataFrame üzerinde toplama işlemleri yapma"""
        agg_func = params.get("agg_func", "mean")
        columns = params.get("columns")
        
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            # Sütunların varlığını kontrol et
            valid_columns = [col for col in columns if col in self.df.columns]
            if not valid_columns:
                return self.df
                
            return self.df[valid_columns].agg(agg_func).to_frame().reset_index()
        else:
            # Sayısal sütunları otomatik kullan
            return self.df[self.numeric_columns].agg(agg_func).to_frame().reset_index()

    def transform_pivot(self, params: Dict) -> pd.DataFrame:
        """Pivot tablo oluşturma"""
        index = params.get("index")
        columns = params.get("columns")
        values = params.get("values")
        
        if not index or not columns or not values:
            return self.df
            
        # Sütunların varlığını kontrol et
        if index not in self.df.columns or columns not in self.df.columns or values not in self.df.columns:
            return self.df
            
        try:
            return self.df.pivot_table(index=index, columns=columns, values=values, aggfunc='mean').reset_index()
        except:
            # Pivot başarısız olursa orijinal veriyi döndür
            return self.df
