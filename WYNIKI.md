# Wyniki FD-DINOv2

Pomiar: `fid_eval.py` / `train.py --fid-every` (`aurora/fid.py`). 4096 promptów z walidacji (pierwsza caption każdego
z 4096 obrazów z 4 shardów walidacyjnych), 50 kroków Eulera, CFG 3.0, seed 0, cechy CLS DINOv2-B/14 z rejestrami
(`vit_base_patch14_reg4_dinov2.lvd142m`), referencja = prawdziwe latenty walidacyjne przez ten sam dekoder FLUX.2.
Niżej = lepiej. Liczby porównywalne tylko w obrębie tej konfiguracji (nie z FD-DINOv2 z publikacji: inny enkoder,
inny zbiór, 4096 zamiast 50k próbek). Pomiar w pętli i offline zgadzają się (36.53 vs 36.51 dla kroku 48,128).

## Wydane modele

| model | denoiser | obrazy w treningu | wagi | FD-DINOv2 |
|---|---|---|---|---|
| IMG-50M (`runs/muloco50m_2026_10_30_1/merged.pt`) | 50M, TE 13m od zera | 63.4M | WSM | 69.61 |
| IMG-253M (`runs/253m_slayer_repa/latest.pt`) | 253M, Slayer149 | 29.4M | EMA | 37.64 |
| IMG-641M-epoch10.8 (`runs/2560x6_hero`, krok 62,720) | 641M (2560×6), Slayer149 + refiner 1280 | 63.3M | EMA | 29.50 |

## Hero run 2560×6 (`runs/2560x6_hero`, 5.88M obrazów na epokę)

Odcinki (wznowienia z checkpointu, dane kontynuowane od tego samego obrazu):

| kroki | batch | lr Muon | REPA |
|---|---|---|---|
| 0 – 2,112 | 4096 | 1e-2 | iREPA 0.5 każdy krok do 1,435 |
| 2,112 – 34,944 | 512 | 3e-3 | off |
| 34,944 – 45,056 | 1024 | 3e-3 | iREPA 0.5 każdy krok do 44,711 |
| 45,056 – 48,128 | 4096 | 1e-2 | REPA 0.5 co 32 kroki |
| 48,128 – 69,888 | 1024 | 3e-3 | iREPA 0.5 każdy krok |
| 69,888 – 86,997 | 1024 | 3e-3 | off |

| krok | obrazy | epoki | wagi | FD-DINOv2 |
|---|---|---|---|---|
| 4,288 | 9.76M | 1.66 | EMA | 53.54 |
| 8,576 | 11.96M | 2.03 | EMA | 48.25 |
| 12,864 | 14.16M | 2.41 | EMA | 44.82 |
| 17,152 | 16.35M | 2.78 | EMA | 42.28 |
| 21,440 | 18.55M | 3.15 | EMA | 40.17 |
| 25,728 | 20.74M | 3.53 | EMA | 38.55 |
| 30,016 | 22.94M | 3.90 | EMA | 37.05 |
| 34,304 | 25.13M | 4.27 | EMA | 36.18 |
| 34,560 | 25.26M | 4.30 | raw (merge/) | 38.80 |
| 40,704 | 31.36M | 5.33 | EMA | 33.47 |
| 46,080 | 40.01M | 6.80 | EMA | 43.15 |
| 48,128 | 48.40M | 8.23 | EMA | 36.51 |
| 60,928 | 61.51M | 10.46 | EMA | 29.80 |
| 62,720 | 63.34M | 10.77 | EMA | 29.50 |
| 69,632 | 70.42M | 11.98 | EMA | 28.99 |
| 86,997 (koniec) | 88.20M | 15.00 | EMA | 28.27 |
| 86,997 (koniec) | 88.20M | 15.00 | WSM/QAM (`merged.pt`, snapshoty 70,400–86,997) | **27.66** |

Wnioski:
- Skok lr 3e-3 → 1e-2 (krok 45,056) pogorszył FD z 33.5 do 43.2; po powrocie do 3e-3 z iREPA szybki spadek do 29.5.
  `val/loss` prawie tego nie pokazała — przy zmianach lr patrzeć na FD.
- iREPA przy lr 3e-3 nie szkodziła w środku treningu (wbrew HASTE), raczej pomogła (40,704: ~0.5 lepiej niż trend
  bez REPA). Około kroku 69k strata `repa` się nasyciła (-0.735) i REPA została wyłączona.
- EMA ~2.6 punktu lepsze od surowych wag (34,304 vs 34,560).
- Na końcu WSM/QAM z 6 snapshotów ostatniego odcinka (bez REPA) lepszy od EMA: 27.66 vs 28.27.
- Po wyłączeniu REPA (69,888) EMA `val/loss` spadała ~4× szybciej niż w ostatnim odcinku z iREPA.
- Przy podobnej liczbie obrazów: 641M ~33.5 (31.4M obrazów) vs IMG-253M 37.6 (29.4M); IMG-50M przy 63.4M obrazów 69.6.

EMA `val/loss` (ten sam zbiór, stałe t i szum): 0.7107 (1,536) → 0.6531 (34,816) → 0.6431 (49,152) → 0.6378 (61,440)
→ 0.6365 (69,632) → 0.6327 (75,776) → 0.6303 (86,997). Dla porównania IMG-50M (EMA) 0.7030.
