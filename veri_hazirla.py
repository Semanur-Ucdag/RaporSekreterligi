import os
import json

# Klasör yolları
ses_klasoru = r"C:\Users\sema.nur\Desktop\STTdememe\kayitlar-1"
rapor_klasoru = r"C:\Users\sema.nur\Desktop\STTdememe\raporlar"
json_kayit_yolu = r"C:\Users\sema.nur\Desktop\STTdememe\hazir_dataset.json"

# 1. Adım: Tüm raporları UTF-8'e dönüştür
print("🔄 Rapor dosyaları UTF-8'e dönüştürülüyor...\n")
for dosya_adi in os.listdir(rapor_klasoru):
    if dosya_adi.endswith(".txt"):
        tam_yol = os.path.join(rapor_klasoru, dosya_adi)
        try:
            # İlk olarak cp1254 (Türkçe ANSI) ile açmayı dene
            with open(tam_yol, "r", encoding="cp1254") as f:
                icerik = f.read()
            # UTF-8 olarak yeniden kaydet
            with open(tam_yol, "w", encoding="utf-8") as f:
                f.write(icerik)
            print(f"✅ Dönüştürüldü: {dosya_adi}")
        except Exception as e:
            print(f"⚠️ Atlandı (zaten UTF-8 olabilir): {dosya_adi} – {e}")

# 2. Adım: Dataset oluştur
print("\n📦 Dataset oluşturuluyor...\n")
dataset = []

for dosya_adi in os.listdir(ses_klasoru):
    if dosya_adi.endswith(".wav"):
        dosya_adi_no_uzanti = os.path.splitext(dosya_adi)[0]
        ses_yolu = os.path.join(ses_klasoru, dosya_adi)
        rapor_yolu = os.path.join(rapor_klasoru, dosya_adi_no_uzanti + ".txt")

        if os.path.exists(rapor_yolu):
            try:
                with open(rapor_yolu, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                dataset.append({
                    "path": ses_yolu,
                    "transcription": transcript
                })
                print(f"✅ Eklendi: {dosya_adi}")
            except Exception as e:
                print(f"⛔ Hata okuma sırasında ({dosya_adi}): {e}")
        else:
            print(f"⚠️ Transkript bulunamadı: {dosya_adi_no_uzanti}.txt")

# 3. Adım: JSON'a yaz
with open(json_kayit_yolu, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"\n✅ Dataset hazırlandı ve şurada kaydedildi: {json_kayit_yolu}")
print(f"📊 Toplam {len(dataset)} kayıt eklendi.")
print("🏁 İşlem tamamlandı.")
