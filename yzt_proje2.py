import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt # 'matplotlib' -> 'matplotlib.pyplot as plt' olarak duzeltildi

# CSV dosyanizin adini ve yolunu buraya yazin
dosya_yolu = 'movies_initial.csv' 

# Gerekli sutunlar
kullanilacak_sutunlar = [
    'title', 
    'year', 
    'runtime', 
    'genre', 
    'director',
    'imdbRating',
    'imdbVotes'
]

print("--- Asama 1: Veri Yukleme ve Doldurma Basliyor ---")

try:
    df = pd.read_csv(
        dosya_yolu, 
        usecols=kullanilacak_sutunlar,
        encoding='utf-8'
    )
    
    df_temiz = df.copy()
    print("Veri ham olarak yuklendi.")

    # --- TEMIZLEME ve DOLDURMA ADIMLARI ---

    # 1. 'runtime' (Sure) Temizleme ve Doldurma
    print("\n1. 'runtime' sutunu isleniyor...")
    
    # Adim 1a: "1 min" formatini "1" (sayi) formatina cevir
    df_temiz['runtime'] = df_temiz['runtime'].str.replace(' min', '')
    
    # Adim 1b: Sutunu sayisal (numeric) hale getir
    df_temiz['runtime'] = pd.to_numeric(df_temiz['runtime'], errors='coerce')
    
    # Adim 1c: Ortalamayi HESAPLA
    ortalama_sure = df_temiz['runtime'].mean()
    
    # Adim 1d: Eksik (NaN) degerleri bu ortalama ile DOLDUR
    df_temiz['runtime'] = df_temiz['runtime'].fillna(ortalama_sure)
    
    # Adim 1e: Sureleri tam sayiya (Integer) cevir
    df_temiz['runtime'] = df_temiz['runtime'].astype(int)
    
    print(f"'runtime' eksik degerleri {int(ortalama_sure)} (ortalama) ile dolduruldu.")


    # 2. 'imdbRating' (Puan) Temizleme ve Doldurma
    print("\n2. 'imdbRating' sutunu isleniyor...")
    
    # Adim 2a: Sutunu sayisal (numeric) hale getir
    df_temiz['imdbRating'] = pd.to_numeric(df_temiz['imdbRating'], errors='coerce')
    
    # Adim 2b: Ortalamayi HESAPLA
    ortalama_puan = df_temiz['imdbRating'].mean()
    
    # Adim 2c: Eksik (NaN) degerleri bu ortalama ile DOLDUR
    df_temiz['imdbRating'] = df_temiz['imdbRating'].fillna(ortalama_puan)
    
    print(f"'imdbRating' eksik degerleri {ortalama_puan:.2f} (ortalama) ile dolduruldu.")


    # 3. Kritik Kategorik Verilerin Kontrolu
    print("\n3. Kritik kategorik (metin) veriler kontrol ediliyor...")
    df_temiz = df_temiz.dropna(subset=['director', 'genre'])
    print("'director' veya 'genre' bilgisi hala eksik olan satirlar silindi.")


    # 4. 'year' (Yil) Sutununu Sayiya Cevirme (YENI EKLENDI)
    # Sonraki analizler icin yili sayi yapmak zorundayiz
    print("\n4. 'year' sutunu temizleniyor...")
    df_temiz['year'] = pd.to_numeric(df_temiz['year'], errors='coerce')
    df_temiz = df_temiz.dropna(subset=['year']) # Yili olmayanlari sil
    df_temiz['year'] = df_temiz['year'].astype(int)
    print("'year' sutunu sayiya donusturuldu.")


    # 5. BOZUK KARAKTERLERI SILME (YENI EKLENDI - Print Hatasi Cozumu)
    print("\n5. Yazdirilamayan bozuk karakterler temizleniyor...")
    metin_sutunlari = df_temiz.select_dtypes(include=['object']).columns

    for sutun in metin_sutunlari:
        # Standart olmayan (ASCII disi) karakterleri silip atar
        df_temiz[sutun] = df_temiz[sutun].astype(str).apply(
            lambda x: x.encode('ascii', 'ignore').decode('utf-8')
        )
    print("Bozuk karakter temizligi tamamlandi.")

    
    # --- SON KONTROL ---
    print("\n--- ASAMA 1 BASARIYLA TAMAMLANDI! ---")
    print("Veri Temizlendi ve Dolduruldu.\n")
    
    print("Temizlenmis Veri Ozeti (info):")
    df_temiz.info()
    
    print("\nTemizlenmis Veri (Ilk 5 Satir):")
    # Artik burasi hata vermeyecek
    print(df_temiz.head())

    # ----------------------------------------------------
    # --- ASAMA 2: KESIFSEL VERI ANALIZI (EDA) BASLIYOR ---
    # ----------------------------------------------------
    print("\n\n--- ASAMA 2: Kesifsel Veri Analizi ---")

    # 1. 'genre' (Tur) Sutununu Ayristirma
    # "Drama, Crime" -> iki ayri satir ('Drama' ve 'Crime') yapar.
    # Bu, tur analizleri icin COK onemslidir.
    print("\n1. 'genre' sutunu analiz icin ayristiriliyor...")
    
    # 'genre' sutununu virgul ve bosluga gore ayir
    df_genres_exploded = df_temiz.assign(genre=df_temiz['genre'].str.split(', '))
    
    # 'explode' ile her bir turu kendi satirina kopyala
    df_genres_exploded = df_genres_exploded.explode('genre')
    
    # ' Short' gibi olasi bas/son bosluklarini temizle
    df_genres_exploded['genre'] = df_genres_exploded['genre'].str.strip()
    
    print("'genre' sutunu basariyla ayristirildi.")
    print("Ayristirilmis veri ornegi (ilk 5 satir):")
    print(df_genres_exploded.head())


    # --- ANALIZ 1: EN POPULER TURLER (TUM ZAMANLAR) ---
    print("\n--- Analiz 1: En Populer 10 Tur (Tum Zamanlar) ---")
    
    # 'genre' sutunundaki her bir turun kac kez gectigini say
    populer_turler = df_genres_exploded['genre'].value_counts().head(10)
    print(populer_turler)
    
    # --- ANALIZ 2A: EN POPULER YONETMENLER (TUM ZAMANLAR) ---
    print("\n--- Analiz 2A: En Populer 10 Yonetmen (Tum Zamanlar) ---")
    
    # Yonetmen analizleri icin 'df_temiz' (orijinal temiz veri) kullanilir,
    # cunku 'df_genres_exploded' verisi film sayisini yanlis gosterir (sisirir).
    
    # .value_counts() her bir yonetmenin adini sayar ve en coktan aza dogru siralar.
    populer_yonetmenler_tum_zamanlar = df_temiz['director'].value_counts().head(10)
    
    print("Tum zamanlarda en cok filme sahip 10 yonetmen:")
    print(populer_yonetmenler_tum_zamanlar)


    # --- ANALIZ 2B: YAKIN ZAMANDAKI (2010 SONRASI) EN POPULER YONETMENLER ---
    print("\n--- Analiz 2B: En Populer 10 Yonetmen (2010 Sonrasi) ---")
    print("Bu analiz, 'mevcut yonelimi' gosterir.")
    
    # 'year' sutununu 2010'dan buyuk veya esit olan filmleri filtrele
    df_recent_films = df_temiz[df_temiz['year'] >= 2010]
    
    # Simdi bu YENI (filtrelenmis) veri uzerinden sayim yap
    populer_yonetmenler_recent = df_recent_films['director'].value_counts().head(10)
    
    print("2010 sonrasi en cok filme sahip 10 yonetmen:")
    print(populer_yonetmenler_recent)

    # --- ANALIZ 3A: EN BASARILI YONETMENLER (TUM ZAMANLAR - PUANA GORE) ---
    print("\n--- Analiz 3A: En Basarili 10 Yonetmen (Tum Zamanlar - Puana Gore) ---")
    
    # 'Tutarli basariyi' olcmek icin bir esik degeri (minimum film sayisi) belirliyoruz.
    # Bu degeri 3, 5, 10 gibi degistirebilirsin.
    esik_degeri_all_time = 7 
    
    # Adim 1: Her yonetmenin toplam film sayisini bul
    director_film_counts_all = df_temiz['director'].value_counts()
    
    # Adim 2: Sadece esik degerinden fazla film yapmis yonetmenlerin listesini al
    prolific_directors_all = director_film_counts_all[director_film_counts_all >= esik_degeri_all_time].index
    
    # Adim 3: Ana veriyi (df_temiz) bu "uretken" yonetmenlere gore filtrele
    df_prolific_all = df_temiz[df_temiz['director'].isin(prolific_directors_all)]
    
    # Adim 4: Simdi bu filtrelenmis grup icinde puan ortalamasini al
    basarili_yonetmenler_all = df_prolific_all.groupby('director')['imdbRating'].mean().sort_values(ascending=False).head(10)
    
    print(f"En az {esik_degeri_all_time} film yapmis en basarili 10 yonetmen (Tum Zamanlar):")
    print(basarili_yonetmenler_all)


    # --- ANALIZ 3B: EN BASARILI YONETMENLER (2010 SONRASI - PUANA GORE) ---
    print("\n--- Analiz 3B: En Basarili 10 Yonetmen (2010 Sonrasi - Puana Gore) ---")
    
    # 2010 sonrasi veri seti daha kucuk oldugu icin esik degerini dusuruyoruz
    esik_degeri_recent = 3 
    
    # 'df_recent_films' verisini (Analiz 2B'de olusturduk) kullaniyoruz.
    
    # Adim 1: 2010 sonrasi film sayilarini bul
    director_film_counts_recent = df_recent_films['director'].value_counts()
    
    # Adim 2: 2010 sonrasi esik degerini gecen yonetmenleri bul
    prolific_directors_recent = director_film_counts_recent[director_film_counts_recent >= esik_degeri_recent].index
    
    # Adim 3: 2010 sonrasi veriyi bu yonetmenlere gore filtrele
    df_prolific_recent_basari = df_recent_films[df_recent_films['director'].isin(prolific_directors_recent)]
    
    # Adim 4: Ortalama puani hesapla
    basarili_yonetmenler_recent = df_prolific_recent_basari.groupby('director')['imdbRating'].mean().sort_values(ascending=False).head(10)

    print(f"2010 sonrasi en az {esik_degeri_recent} film yapmis en basarili 10 yonetmen (Mevcut Yonelim):")
    print(basarili_yonetmenler_recent)

    # --- GORSEL 2: En Populer Yonetmenler (Tum Zamanlar - Film Sayisi) ---
    print("Gorsel 2 hazirlaniyor: En Populer Yonetmenler (Tum Zamanlar)...")

    # Veri: 'populer_yonetmenler_tum_zamanlar' (Analiz 2A'da hesaplanmisti)

    plt.figure(figsize=(12, 8)) # Grafik boyutu (genislik, yukseklik)

    # 'kind='barh'' -> Yatay Bar Grafik cizer.
    # En yuksek sayida olani en uste almak icin '.sort_values(ascending=True)'
    populer_yonetmenler_tum_zamanlar.sort_values(ascending=True).plot(
        kind='barh', 
        color='darkcyan', 
        edgecolor='black'
    )
    
    plt.title('En Populer 10 Yonetmen (Tum Zamanlar - Film Sayisi)', fontsize=16)
    plt.xlabel('Toplam Film Sayisi', fontsize=12)
    plt.ylabel('Yonetmen (Director)', fontsize=12)
    plt.tight_layout() # Kenar bosluklarini ayarla
    
    print("Grafik gosteriliyor...")
    plt.show()


    # --- GORSEL 3: En Populer Yonetmenler (2010 Sonrasi - Film Sayisi) ---
    print("Gorsel 3 hazirlaniyor: En Populer Yonetmenler (2010 Sonrasi)...")

    # Veri: 'populer_yonetmenler_recent' (Analiz 2B'de hesaplanmisti)

    plt.figure(figsize=(12, 8)) # Grafik boyutu

    # 'color='goldenrod'' (Altin rengi) -> Renkleri degistirebiliriz
    populer_yonetmenler_recent.sort_values(ascending=True).plot(
        kind='barh', 
        color='goldenrod', 
        edgecolor='black'
    )
    
    plt.title('En Populer 10 Yonetmen (2010 Sonrasi - Film Sayisi)', fontsize=16)
    plt.xlabel('Toplam Film Sayisi (2010 Sonrasi)', fontsize=12)
    plt.ylabel('Yonetmen (Director)', fontsize=12)
    plt.tight_layout()
    
    print("Grafik gosteriliyor...")
    plt.show()

    # --- ANALIZ 4: GUNCEL TUR EGILIMI (2010 SONRASI) ---
    print("\n--- Analiz 4: Guncel Tur Egilimi (2010 Sonrasi) ---")
    
    # 'df_genres_exploded' verisini (Analiz 1'de olusturduk) kullaniyoruz
    # cunku bu veri turleri ayristirilmisti.
    
    # Sadece 2010 ve sonrasi filmleri iceren bir veri seti olustur:
    df_recent_genres = df_genres_exploded[df_genres_exploded['year'] >= 2010]


    # --- Analiz 4A: Guncel Populerlik (Uretim Hacmi) ---
    print("\n--- Analiz 4A: Guncel Tur Egilimi (Populerlik/Uretim Hacmi) ---")
    
    recent_genre_counts = df_recent_genres['genre'].value_counts().head(10)
    
    print("2010 Sonrasi En Cok Uretilen (Populer) 10 Tur:")
    print(recent_genre_counts)
    

    # --- Analiz 4B: Guncel Basari (Puan Ortalamasi) ---
    print("\n--- Analiz 4B: Guncel Tur Egilimi (Basari/Puan Ortalamasi) ---")
    
    # Yine 'tek filmlik harikalari' onlemek icin bir esik degeri koyuyoruz.
    # 2010 sonrasi en az 10 kez listede gorunmus turleri filtreleyelim.
    esik_degeri_tur = 10 
    
    # Adim 1: 2010 sonrasi her turun film sayisini (tamamini) bul
    all_recent_genre_counts = df_recent_genres['genre'].value_counts()
    
    # Adim 2: Esik degerini gecen (stabil) turleri sec
    stable_genres = all_recent_genre_counts[all_recent_genre_counts >= esik_degeri_tur].index
    
    # Adim 3: Veriyi bu 'stabil' turlere gore filtrele
    df_recent_stable_genres = df_recent_genres[df_recent_genres['genre'].isin(stable_genres)]
    
    # Adim 4: Simdi puan ortalamasini al
    recent_genre_ratings = df_recent_stable_genres.groupby('genre')['imdbRating'].mean().sort_values(ascending=False).head(10)
    
    print(f"2010 Sonrasi En Basarili 10 Tur (En az {esik_degeri_tur} film):")
    print(recent_genre_ratings)

    # --- ANALIZ 5: MAKINE OGRENMESI YAKLASIMI (EWMA) ILE TUR EGILIMI ---
    print("\n--- Analiz 5: Agirlikli Tur Egilimi (EWMA Algoritmasi) ---")
    print("Bu analiz, son yillara daha fazla agirlik verir.")

    # Adim 1: Her tur icin yillara gore uretim sayisini hesapla
    # Cikti: (Yil, Tur) -> Sayi
    print("Yillara gore tur sayimlari hazirlaniyor...")
    genre_yearly_counts = df_genres_exploded.groupby(['year', 'genre']).size()

    # Adim 2: Veriyi 'pivot' (matris) haline getir
    # Indeksler: Yillar (or. 1990, 1991...)
    # Sutunlar: Turler (or. Action, Comedy, Drama...)
    # Hucreler: O yil o turden kac film ciktigi
    genre_yearly_pivot = genre_yearly_counts.unstack(fill_value=0)

    # Adim 3: EWMA (Ussel Agirlikli Hareketli Ortalama) "Algoritmasini" Uygula
    # 'span=10', son 10 yilin verisine daha fazla agirlik veren bir ayardir.
    # Bu degeri (or. span=5) degistirerek agirligi artirabilirsin.
    print(f"EWMA (span=10) algoritmasi uygulanarak egilim hesaplaniyor...")
    genre_ewm_trend = genre_yearly_pivot.ewm(span=10, adjust=False).mean()

    # Adim 4: Guncel (En Son) Egilim Puanlarini Al
    # 'genre_ewm_trend' tablosunun en son satirini aliriz (en guncel yil)
    current_trend_scores = genre_ewm_trend.iloc[-1]

    # Adim 5: Sonucu Goster
    print("\n--- GUNCEL EGILIM PUANLARI (En Yuksek 15 Tur) ---")
    print("Puan ne kadar yuksekse, o turun guncel egilimi o kadar gucludur.")
    print(current_trend_scores.sort_values(ascending=False).head(15))

    # --- ASAMA 3: VERI GORSELLESTIRME BASLIYOR ---
    # ----------------------------------------------------
    print("\n\n--- ASAMA 3: Veri Gorsellestirme ---")
    print("Gorsel 1 hazirlaniyor: Guncel Tur Egilimi (EWMA Puanlari)...")

    # --- GORSEL 1: Guncel Tur Egilimi (Bar Grafigi) ---
    # Analiz 5'in ciktisini (current_trend_scores) gorsellestiriyoruz.
    
    # Gorsellestirmek icin ilk 15 turu alalim
    gorsel_veri_1 = current_trend_scores.sort_values(ascending=False).head(15)

    # Cizim boyutunu ayarla (genislik, yukseklik)
    plt.figure(figsize=(14, 7))
    
    # Pandas'in .plot() fonksiyonunu kullanarak bar grafigi olustur
    # 'skyblue' yerine baska bir renk de secebilirsin, or. 'darkblue'
    gorsel_veri_1.plot(kind='bar', color='skyblue', edgecolor='black')
    
    # Baslik ve Eksen Isimleri (Ingilizce karakterlerle)
    plt.title('Guncel Tur Egilimi (EWMA Agirlikli Puanlar)', fontsize=16)
    plt.ylabel('Agirlikli Egilim Puani (EWMA Score)', fontsize=12)
    plt.xlabel('Tur (Genre)', fontsize=12)
    
    # X eksenindeki tur isimlerini 45 derece dondur (sigmasi icin)
    plt.xticks(rotation=45, ha='right')
    
    # Grafigin kenar bosluklarini ayarla (yazilarin kesilmemesi icin)
    plt.tight_layout()
    
    # Grafigi ekranda goster
    print("Grafik gosteriliyor...")
    plt.show()

    

except FileNotFoundError:
    print(f"Hata: '{dosya_yolu}' adinda bir dosya bulunamadi.")
except Exception as e:
    print(f"Bir hata olustu: {e}")