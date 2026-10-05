# Adım 21 — İnsan testi: model, bir insana (Görkan) göre ne kadar iyi? PLAN (2026-10-05, Görkan ONAYLADI)

## 0. Neden
- Adım 15-20'nin bütün altınları Claude etiketi. "Kazandı" hep "Claude'un kurallarına göre okumada kazandı" demekti.
- İnsanla elimizdeki tek ölçüm kalibrasyon 100 (Adım 18): Görkan–Claude çift F1 **0.64**, konu F1 0.74, ortak konuda
  duygu uyumu 0.87-0.90. Ama o 100 yorum Adım 19'da EĞİTİME girdi → modeli onunla ölçemeyiz.
- Bu adım **model eğitmez**. Soru: **dondurulmuş uygulama hattı (Adım 19 konu + Adım 17 duygu), Görkan'ın okumasına
  bir Claude etiketleyici kadar yakın mı?**

## 1. Veri — 100 yeni yorum (id 9500-9599)
- Aynı tarif: 8-40 kelime, hiç görülmemiş, `check_no_leak` + Adım 16-19'un bütün setleriyle çakışma assert'i.
- 70 pozitif + 30 negatif havuz etiketli (negatif havuz ~250; 30 harcar, ~220 kalır). Gerekçe: negatif havuz azalıyor;
  rapor iki havuzu ayrı da verir. (Önceki setler 50/50 idi — bu fark rapora yazılır.)
- Karışık sıra, 4 dosya × 25 (`human_gorkan_01..04.csv`: no, id, text, boş `etiket`) — Adım 18'deki biçim.

## 2. Etiketler (hepsi kör, birbirini görmeden)
1. **Görkan:** 100 yorum, `step18_big_test/GORKAN_YONERGE.md` aynen (~50 dk, 4 parça, istediği zaman).
   Önce commit'lenir. Görkan model çıktısına ve Claude etiketine bakmaz.
2. **Taze oturum (fresh):** aynı 100, sadece LABEL_RULES.md — "Claude etiketleyici" referansı. (Kayma kontrolü
   gerekmez: Adım 19'da 0.918 ölçüldü.)
3. **Model:** dondurulmuş uygulama hattı (`step9_app/aspect_pipeline.py`, 3 tohum), Mac mini'de bir kez. Konu başına
   nötr üretmez (bilinen sınırlama).

## 3. Ölçüm (önceden sabit)
- **ANA:** (konu, duygu) çift F1, **model–Görkan** ile **fresh–Görkan** farkı; yorum bazında eşleştirilmiş bootstrap
  (2000, tohum 21). Okuma: aralık 0'ı içeriyorsa "model, Görkan'a bir Claude etiketleyici kadar yakın; fark ayırt
  edilemiyor"; altında 0'ı dışlıyorsa "model Claude etiketleyiciden geride".
- İkincil (hepsi Görkan'ı referans alarak):
  - Görkan'ın konularını **bulma oranı** (recall) ve modelin konularının Görkan'da olma oranı (precision) — iki
    taraf için; Adım 18'de Claude, Görkan'dan %36 fazla etiket veriyordu.
  - **Ortak konularda duygu uyumu** (kullanıcı için en anlaşılır sayı: "ikimiz de kargo dedik, duyguda anlaştık mı?").
  - Konu başına kappa; nötr hariç çift F1 (model nötr üretemediği için adil kıyas).
  - Model–fresh çift F1 (modelin kendi etiketleyici ailesine uyumu; kasa testindeki 0.76'yla tutarlı mı).
  - Havuza göre (pozitif / negatif) ayrı.
- **Güç (dürüst):** n=100 ile çift F1 farkının aralığı ~±0.06. Küçük farklar ayırt edilemez; bu bir "kabaca nerede
  duruyoruz" ölçümü.

## 4. Beklenti (önceden)
- fresh–Görkan çift F1 0.60-0.68 (kalibrasyonda 0.64).
- model–Görkan çift F1 0.55-0.65; fark (model − fresh) −0.08 ile 0; 0'ı dışlama ~yarı yarıya.
- Ortak konuda duygu uyumu: fresh ~0.87, model ~0.85. Model–fresh çift F1 ~0.72-0.80.
- Konu kapsamı: hem model hem fresh Görkan'dan ~%25-40 fazla konu verir (Q/P en büyük ayrışma).

## 5. Sonra
- Sonuç toplu sayılarla raporlanır. **Ardından (isteğe bağlı, Görkan'la):** ayrışan 10-15 yorum birlikte okunur —
  bu set o andan sonra "okunmuş" olur ve ileride eğitime girebilir. Okuma, ölçümden SONRA ve rapor yazıldıktan sonra.
- Kural/model değişikliği bu adımın kapsamı dışında; çıkan fikirler PROGRESS'e "sonraki adım için" diye yazılır.

## 6. Kapsam dışı
Eğitim, eşik/ayar, LABEL_RULES değişikliği, Görkan'ın Claude etiketlerini düzeltmesi/uzlaştırma.

## Sıra ve duraklar
1. (onay) → 100 yorum seçimi + sızıntı assert'leri + Görkan'ın 4 dosyası → commit (laptop).
2. Görkan 100'ü etiketler (istediği zaman) → commit.
3. Fresh oturum aynı 100'ü etiketler → `fresh-step21` dalı → commit. Model tahminleri Mac mini'de → commit.
4. `evaluate21.py` → rapor → **DUR** → (isteğe bağlı) ayrışmaları birlikte okuma → PROGRESS.

## Görkan'a düşen iş
~50 dk kör etiketleme (4 × ~12 dk; dosyayı açmak istemezsen sohbete `1: Kp,Qn` biçiminde yazman da olur) +
isteğe bağlı ~15 dk ayrışma konuşması. Mac mini oturumlarında commit/push onayları.
