# Replikacja AuroraIMG2-50M-Preview na 8× RTX 3090 Ti

Cel: wytrenować od zera ten sam przepis (text encoder 12.6M + DiT, flow matching na latentach FLUX.2 256px)
z hiperparametrami z README. **Na start bierzemy wersję 116M** (`--size 116m`: DiT 768×12 = 103.7M + text encoder 12.6M).
Wersja `--size 50m` to dokładnie opublikowany model, przydatna do porównania 1:1.

## 1. Co wiemy o oryginale (README + `pipeline.py`)

| | |
|---|---|
| Architektura | dokładnie z `pipeline.py`: odtworzona w `aurora/model.py`, **ładuje opublikowane wagi strict i daje bit-w-bit ten sam output** (`tests/check_model.py`) |
| Dane | `QLNI/FLUX-Reason-6M-flux2-latents`: ~1000 shardów `.npz` w 3 podzbiorach (aesthetics-part01/02, imaginative; podzbioru „text” z karty datasetu na HF jeszcze nie ma), ~385 GB, ~5.6M obrazów, latenty fp16 `(128,16,16)`, już znormalizowane BN i spakowane 2×2 |
| Cel | velocity `eps - z`, `z_t = (1-t)z + t·eps`, t ~ logit-normal(0,1), MSE |
| CFG | 10% captionów → `[BOS][NULL][EOS]` |
| MLM aux | waga 0.05, co 4. krok, 15% maskowania, głowica związana z embeddingiem |
| Optymalizacja (oryginał, u nas `--optimizer muon`) | Muon lr 1e-2 / mom 0.95 / 5× NS na macierzach ukrytych; AdamW (0.9, 0.95): embedding 5e-4, reszta 3e-4; WD 0.01 na Muon + in/out proj |
| Harmonogram (oryginał) | warmup 2%, cosine do 0, clip 1.0, bf16 autocast, `torch.compile`. **U nas domyślnie: warmup 2% + stały lr** |
| EMA | 0.9995 z warmupem, wydawane są wagi EMA |
| Budżet | 247 805 kroków × batch 256 = 63.4M próbek (~11 epok), 10 h na 1× RTX PRO 6000 |

### Czego README nie podaje (moje założenia, łatwe do zmiany)
- **Inicjalizacja**: normal(std = fan_in^-0.5); zera na `ada`, `mod_table`, wyjściach rezydualnych
  (`attn.out`, `cross.out`, `mlp.out_proj`) i `patch_out`. Wtedy „każdy blok startuje jako identyczność”, jak w README.
- **Muon**: wariant Kellera Jordana (Nesterov, skala `sqrt(max(1, rows/cols))`, decoupled WD). Fused `qkv` traktowane jako jedna macierz.
- **Podział AdamW**: `patch_in`/`patch_out`/`time_mlp`/`ada` = „input/output projections” (z WD),
  normy / `mod_table` / biasy bez WD.
- **MLM**: maskujemy tylko zwykłe tokeny (id > 5), zawsze `[MASK]` (bez 80/10/10), na captionach przed CFG-dropem.
- **Tokenizer**: używamy opublikowanego `tokenizer.json`. Był trenowany na tych samych captionach, więc to ten sam
  wynik co trening własnego BPE, a prościej.
- **Hold-out**: ostatni shard każdego podzbioru (obecnie 3 shardy), z nich 4096 obrazów do val. Te same shardy są zawsze
  pobierane, także przy `--fraction`, więc `val/loss` z treningu na 10% i na całości można porównać.
- **EMA warmup**: `decay = min(0.9995, (1+step)/(10+step))`.

## 2. Uproszczenia pod 8× 3090 Ti

**Muon na 8 kartach: zwykły DDP + Muon dzielony między ranki.** Model mieści się w całości na każdej karcie, więc nie ma FSDP/ZeRO:
- DDP all-reduce'uje gradienty. Potem, dla każdej grupy macierzy o tym samym kształcie: każdy rank aktualizuje momentum
  wszystkich (tanie, stan optymalizatora jest identyczny wszędzie, więc wystarcza checkpoint z ranku 0), **liczy Newtona-Schulza
  tylko dla swojego kawałka grupy, w jednym batchowanym wywołaniu**, a aktualizacje w bf16 są zbierane przez all_gather.
- Pierwsza wersja (każdy rank liczy wszystko, macierz po macierzy) była błędem: pomiar pokazał **~40 ms/krok, czyli tyle co
  sam fwd+bwd** przy micro-batchu 32. Po batchowaniu wychodzi 20 ms, po podziale na 8 ranków ~4.5 ms (model 50M, RTX 5080).
- Poprawność: `tests/check_muon.py` pokazuje, że wersja dzielona różni się od zwykłego Muona mniej niż sam bf16 od fp32
  (1.9% vs 2.4%; z NS w fp32 różnica wynosi 0.13%). Wagi na wszystkich rankach są bit w bit identyczne.
- **all-reduce w bf16** (`--grad-comm bf16`, domyślnie): 3090 Ti nie mają P2P, więc gradienty idą przez PCIe/host.
  bf16 zmniejsza ruch z ~252 MB do ~126 MB na krok. Przy podejrzeniu, że to szkodzi: `--grad-comm fp32`.
- **Domyślnie: model `253m` (DiT 1024×16, 240.8M + TE 12.6M), batch 2048 = 8 GPU × 256 bez akumulacji.** Zamiast podbijać batch
  do 8192 (ryzyko przekroczenia krytycznego batcha) inwestujemy w większy model. Komunikacja na próbkę jest taka sama jak przy
  64 × 4, szersze matmule powinny podnieść MFU, a lr macierzy ukrytych skaluje się μP-owo jak 1/szerokość względem 768
  (`1e-4 × 768/1024 = 7.5e-5` przy batchu 256, ×2.83 → 2.12e-4). 367 GFLOP na próbkę.
- **Text encoder 27M (domyślnie, `--te-size 27m`): 512 × 8, 8 głowic (2 KV), MLP 1408.** Rośnie mniej więcej tak samo jak
  denoiser (12.6M → 26.7M wobec 104M → 245M). Słabością tych modeli jest rozumienie promptu (liczenie, położenie, czasowniki),
  a encoder przetwarza tylko 64 tokeny, więc kosztuje ~2% FLOP-ów (375 vs 367 GFLOP na próbkę). Cross-attention w DiT
  czyta wektory 512 zamiast 384 (+4.2M parametrów). Eksport zapisuje oba configi, oryginalny `pipeline.py` je wczytuje.
  Checkpointy sprzed tej zmiany nie mają `te_config` i są traktowane jako `13m`. Wznowienie starego runu: `--te-size 13m`.
- Rozważane wcześniej: **batch 8192 = 8 GPU × 256 × akumulacja 4**. Najpierw było 2048 = 8 × 64 × 4: 2292 img/s, 45.8 TFLOPS/GPU,
  czyli MFU ~30% względem 153 TFLOPS (karty z odblokowaną akumulacją fp32). Pamięci zostawało dużo (11 GB przy mb 64), stąd
  micro-batch 256. Ryzyko: 4× mniej kroków (7 744 na całości) i lr ×5.66. Wyżej niż typowy krytyczny batch małych DiT,
  więc trzeba porównać `val/loss` z runem 2048 w funkcji `train/samples` (oś X w wandb). Jeśli wyjdzie gorzej: `--batch 2048
  --micro-batch 256` (bez akumulacji, ta sama komunikacja na próbkę co przy 64 × 4, ale większy micro-batch).
- Wcześniej, pomiar na `adamo-rigga` (6 kart na PCIe 3.0 x4, 2 na x8):
  przy oryginalnym batchu 256 (8 × 32, bez akumulacji) wymiana gradientów zajmowała ~2/3 kroku. Wychodziło 704 img/s
  na 8 kart, czyli 14 TFLOPS/GPU i MFU 18%, a jedna karta bez komunikacji daje ~360 img/s dla 116M. Z akumulacją ×4 all-reduce
  idzie raz na 256 próbek na kartę zamiast 32, więc jego udział spada ~8×.
- Przy zmianie batcha przepis przeskalowuje się automatycznie względem oryginału (batch 256):
  - **kroki**: `247 805 × fraction × 256 / batch`, czyli dalej ~11 epok (przy 8192: 774 przy 10%, 7 744 przy całości);
  - **lr AdamW**: podawane „przy batchu 256” i mnożone przez `√(batch/256)` (reguła √ dla Adama, Malladi i in. 2022).
    Przy 2048 to ×2.83 (ukryte 2.83e-4, spójne z Surjo: 2e-4 przy 1024), przy 8192 ×5.66 (ukryte 5.66e-4, embedding 2.8e-3).
    `--no-lr-batch-scaling` wyłącza. Lr Muona nie jest skalowany;
  - **EMA**: `0.9995^(batch/256)` na krok (0.98412 przy 8192), czyli to samo okno w próbkach co w oryginale;
  - eval co 1/20 runu, `latest.pt` co 1/50, snapshoty co 1/10, WSM co 1/100.
  `--batch 256 --micro-batch 32` przywraca dokładnie oryginalny batch.
- **Dane: zamiast „rotującej puli shardów w VRAM” robimy mmap z dysku.** Oryginał miał 96 GB VRAM, my mamy 24 GB na kartę.
  Shardy są konwertowane do surowych `.npy` (fp16) i czytane przez `np.load(mmap_mode="r")`. Page cache jest współdzielony
  przez wszystkie procesy, losowy dostęp idzie po całym datasecie, a kolejność jest deterministyczna i wznawialna.
  Wymagany odczyt: ~4–5k img/s × 64 KB ≈ 300 MB/s losowych odczytów, co spokojnie wytrzyma NVMe (HDD już nie).
- EMA trzymane tylko na ranku 0 (eval i checkpointy też tam).

### Rozmiar modelu (ile udźwignie DDP)
Pomiary bez `torch.compile`, przeliczone na 24 GB:

| `dim × depth` | parametry (z TE) | pamięć | uwagi |
|---|---|---|---|
| 640 × 8 (`50m`) | 63M | 4.8 GB @ mb 32 | oryginał |
| **768 × 12 (`116m`)** | **116M** | **8.3 GB @ mb 32** | **start** |
| 1024 × 16 | 253M | 11.2 GB @ mb 16 | mb 32 zmieści się (~17 GB) |
| 1280 × 20 | 477M | 12.7 GB @ mb 8 | mb 16 + akumulacja ×2 |
| 1536 × 24 | 802M | OOM na 16 GB | potrzebny ZeRO-1/FSDP |

Komunikacja DDP nie jest tu ograniczeniem: ruch i obliczenia rosną liniowo z liczbą parametrów, więc ich proporcja się nie zmienia.
Ograniczeniem jest pamięć (~16 B/parametr plus aktywacje).

### Optymalizator: domyślnie AdamW (`--optimizer adamw`)
Wszystkie parametry idą przez AdamW. Grupy zostają jak w oryginale: embedding 5e-4, in/out projekcje i normy 3e-4.
Te wartości to AdamW z oryginału **przy tym samym batchu 256**, więc przenoszą się bez zmian. Nowa jest tylko grupa
macierzy ukrytych (dotąd Muon): `--lr-hidden`, domyślnie **1e-4**, `beta2` 0.99.
Oryginalny przepis z Muonem: `--optimizer muon --schedule cosine` (WD wtedy domyślnie 0.01).

**Weight decay 0.1 i stały lr (nasze decyzje, różnią się od oryginału i od Surjo):**
- WD 0.1 (decoupled, `torch.optim.AdamW`) na macierzach ukrytych i in/out projekcjach. Embedding, normy, `mod_table` i biasy są bez WD.
- Po warmupie (2% ≈ 5k kroków) **lr jest stały do końca** (`--schedule constant`). Cosine zostaje jako opcja `--schedule cosine`.
- W PyTorch WD działa jako `p ← p·(1 − lr·wd)`, więc liczy się iloczyn. Skala czasowa zapominania to `τ = 1/(lr·wd)`:
  dla macierzy ukrytych 1/(1e-4·0.1) = **100k kroków (~40% runu)**, dla in/out projekcji (3e-4) ~33k kroków. To rozsądny zakres
  według Wang & Aitchison 2024 (AdamW jako EMA wag). Przy stałym lr τ też jest stałe, więc WD działa równo przez cały trening.
  Uwaga: przy sweepie lr zmienia się też τ. Gdyby trzeba było ruszać lr daleko od 1e-4, warto przeskalować WD, żeby zachować τ.
- Przy stałym lr wagi na końcu nie są „wyżarzone”. Tę rolę pełni EMA, a wypuszczamy właśnie wagi EMA, więc ma ono teraz
  większe znaczenie (patrz niżej). Zaleta: trening można w każdej chwili przedłużyć bez psucia harmonogramu, a `val/loss`
  z krótkich runów sweepu da się porównać uczciwie, bo żaden nie jest w fazie wygaszania.

**Skąd 1e-4.** Za punkt odniesienia biorę Surjo-Image-Preview: 108M, czyli podobna skala, ten sam dataset i latenty.
Ich AdamW: lr 2e-4, betas (0.9, 0.99), bez WD, clip 1.0, 5k kroków warmupu, cosine, **batch 1024**, width 512.
1. **Batch 1024 → 256: lr × √(256/1024) = ×0.5 → 1e-4.** To reguła pierwiastkowa dla Adama (Malladi i in. 2022, wyprowadzona
   z SDE). Daje też `beta2' = 1 − ¼·(1 − 0.99) ≈ 0.9975`; zostawiam 0.99, bo różnica jest mała, a 0.99 jest sprawdzone przez Surjo.
2. **Width 512 → 768:** w μP (dla DiT potwierdzonym w Zheng i in. 2025) lr Adama dla macierzy ukrytych skaluje się jak 1/width,
   co dawałoby ×0.67 → ~7e-5. Ale nasz model jest płytszy (12 vs efektywnie 22 warstwy u Surjo), co pozwala na wyższy lr,
   i nie jest w parametryzacji μP. Te efekty mniej więcej się znoszą, więc zostaję przy 1e-4.
3. **Sprawdzenie krzyżowe:** DiT (Peebles & Xie) trenował wszystkie rozmiary od S do XL na AdamW 1e-4 przy batchu 256.
   To dokładnie nasz batch, więc 1e-4 to sprawdzona, bezpieczna wartość.
4. Warmup się zgadza: nasze 2% × 247 805 = ~5k kroków, tyle samo co u Surjo.

To jest **punkt startowy, a nie optimum**. Optymalny lr małych modeli z QK-norm bywa 2–4× wyżej, dlatego przed pełnym runem
robimy krótki sweep (krok 3b w sekcji 4). Krótkie runy faworyzują wyższe lr, więc przy remisie wybieramy niższy.

**Co tracimy bez Muona:** w LLM-ach Muon daje zwykle ~1.3–2× lepszą efektywność na próbkę. W zamian AdamW jest ~25% szybszy
na krok na tym sprzęcie (116M, 1 GPU: 383 vs 278 img/s; na 8 kartach dochodzi koszt all_gather), prostszy i bez komunikacji
poza DDP. Muon (`aurora/muon.py`, dzielony między ranki, przetestowany) zostaje jako opcja do A/B.

**EMA (do rozważenia):** Surjo używa 0.9999 przy batchu 1024. Przeskalowane na batch 256 (Busbridge i in. 2023: `ρ' = ρ^(1/4)`)
to ~0.99998, czyli bardzo długie okno. Nasze 0.9995 (z oryginału) uśrednia ~2000 kroków (~0.5M próbek). DiT przy batchu 256
używa 0.9999. `val/loss` liczony na EMA pozwoli porównać oba warianty bez nowego treningu, jeśli zapiszemy kilka EMA naraz.
Na razie zostaje 0.9995.

### Wydajność: FlashAttention, TFLOPS, MFU
`train.py` na starcie liczy dokładne FLOP-y kroku (`torch.utils.flop_counter`, fwd + bwd, uśrednione z krokami MLM)
i w logu pokazuje `TFLOPS/GPU` oraz `MFU` (względem szczytu bf16 z akumulacją fp32: 3090 Ti = 80, 4090 = 165, 5080 = 113;
`--peak-tflops` nadpisuje). Koszt na próbkę: **79 GFLOP (`50m`), 160 GFLOP (`116m`)**.

Pomiar na RTX 5080 (116M, mb 32, `torch.compile`): **~62 TFLOPS/GPU, MFU ~55%**. Profil czasu kerneli w jednym kroku:

| część | udział |
|---|---|
| matmule (cuBLAS, ~87% szczytu) | ~60% |
| fuzje Triton z `torch.compile` (RMSNorm, RoPE, modulacja, SwiGLU) | ~17% |
| optymalizator + clip + EMA | ~10–16% |
| attention (SDPA) | ~7% |

- **FlashAttention-2 już jest używany.** PyTorch SDPA sam wybiera wbudowany kernel FA2 (`pytorch_flash::flash_fwd/bwd`)
  dla self-attention w DiT (bez maski, bf16, head_dim 64). Cross-attention i text encoder mają maskę paddingu, więc idą
  przez kernel memory-efficient (cutlass), też fused. Osobny pakiet `flash-attn` mógłby przyspieszyć najwyżej część z tych ~7%,
  więc nie warto go dokładać. FA3 działa tylko na Hopperze.
- Matmule są już blisko szczytu, a resztę skleja `torch.compile`. Wyraźnie więcej dałyby tylko własne kernele, a to się nie opłaca
  przy oczekiwanym zysku rzędu 10–15%.
- Na 3090 Ti szczyt matmuli jest niższy (80 vs 113 TFLOPS), a przepustowość pamięci podobna (~1 TB/s), więc MFU powinno wyjść
  podobne albo wyższe. Spodziewane ~45–55 TFLOPS/GPU, czyli ~300 img/s na kartę dla 116M (przed narzutem komunikacji DDP).

### Trening na części danych: domyślnie 10%
- `--fraction` (w `prepare_data.py convert` i w `train.py`) to zawsze **ułamek pełnego datasetu**, domyślnie **0.1**.
  `convert` zapisuje w katalogu danych `manifest.json` z pełną listą shardów, a `train.py` wybiera z niej ten sam podzbiór.
  Dane pobrane z `--fraction 0.1` i trening z `--fraction 0.1` trafiają więc w te same shardy, bez ryzyka „10% z 10%”.
- 10% to **99 z ~1000 shardów treningowych + 3 walidacyjne: ~39 GB, ~0.57M obrazków**. Shardy są rozłożone równomiernie
  w każdym podzbiorze, więc proporcje danych są takie jak w całości.
- **Liczba kroków** domyślnie skaluje się z ułamkiem i batchem: `247 805 × fraction × 256 / batch`. Przy 0.1 to
  **~11 epok** jak w oryginale: 774 kroki przy batchu 8192 (dla testów na 10% lepiej `--batch 2048 --micro-batch 64`: 3 098 kroków)
  (~6.3M próbek; dla 116M ~45 min na 8× 3090 Ti przy ~2.5k img/s). Log na starcie pokazuje liczbę epok. Przeuczenie widać
  jako rosnący `val/loss`, bo walidacja jest na shardach spoza treningu.
- **Przejście na całość:** `convert --fraction 1` do **tego samego katalogu** (już skonwertowane shardy są pomijane, dociąga się
  tylko brakujące ~90%), potem `tokenize`, potem `train.py --fraction 1` (247 805 kroków). Gdy jakiegoś wybranego sharda brakuje,
  `train.py` kończy się czytelnym błędem z podpowiedzią komendy.

### WSM zamiast wygaszania lr (domyślnie)
Trening idzie ze **stałym lr** (warmup → stała wartość do końca), a efekt wygaszania lr uzyskujemy po fakcie przez
ważone uśrednienie ostatnich checkpointów: **WSM, Warmup-Stable-Merge** ([Tian i in. 2025](https://arxiv.org/abs/2507.17634)).
- Twierdzenie 3.1: średnia checkpointów θ_0..θ_k z wagami `c_0 = 1−w_1`, `c_j = w_j − w_{j+1}`, `c_k = w_k` daje to samo,
  co gdyby kroki między nimi skalować malejącymi współczynnikami w_1..w_k. Każdą krzywą decay da się więc odtworzyć jako średnią.
  Prosta średnia ≙ liniowe wygaszanie.
- W artykule najlepiej wypadło **1-sqrt**, tuż za nim **mean**, a **EMA wyraźnie gorzej**. Najważniejsza jest **długość okna**.
- `train.py` zapisuje surowe wagi (fp32, bez optymalizatora i EMA) do `<out>/merge/step_*.pt` co `--merge-every` kroków
  (domyślnie 1% treningu) przez ostatnie 30% treningu (`--merge-start 0.7`). Wychodzi ~31 plików, **~14 GB dla 116M**
  (~0.46 GB każdy), niezależnie od długości runu. `--merge-every 0` wyłącza zapisy.
- `merge.py <run>` uśrednia je dla każdej kombinacji schematu (`1-sqrt`, `mean`, `cosine`) i okna (5/10/20/30% treningu),
  liczy `val/loss` na hold-oucie, wypisuje porównanie z surowymi wagami i z EMA, a najlepszy wariant zapisuje do `<run>/merged.pt`.
  Dalej `export.py <run>/merged.pt ...` jak zwykle. Okno i schemat dobiera się więc na danych, a nie z góry.
- Ze stałym lr można scalać w dowolnym momencie (np. z ostatnich 10% dotychczasowego treningu), żeby zobaczyć, jak dobry jest
  model teraz, bez przerywania treningu. Trening da się też przedłużyć: nic nie zostało wygaszone.
- EMA (0.9995) dalej liczymy, bo kosztuje mało i daje `val/loss` w logu w trakcie treningu. Na końcu `merge.py` pokazuje, czy
  WSM je bije.

### Checkpointy
- Eval (`val/loss` na EMA) co `--eval-every` kroków, domyślnie **co 1/20 runu**.
- `<out>/latest.pt` co `--ckpt-every` kroków (domyślnie **co 1/50 runu**, przy całości i batchu 8192 co 154 kroki).
  Pełny stan (wagi, optymalizator, EMA, krok), **nadpisywany** atomowo, służy do wznawiania: ten sam `torchrun` z tym samym `--out`.
  Rozmiar ~2.3 GB dla 116M.
- `<out>/snap_<krok>.pt` co `--keep-every` kroków (domyślnie **co 1/10 runu**),
  tylko wagi + EMA (~1 GB dla 116M), **nie są nadpisywane**. Każdy da się wyeksportować:
  `python export.py runs/x/snap_0001545.pt exports/x_half`.
  `--keep-every 0` wyłącza snapshoty.
- `<out>/merge/step_<krok>.pt`: surowe wagi dla WSM, opisane wyżej.
- Wszystko zapisuje rank 0. Na koniec treningu zapis jest zawsze.

### wandb
- Włącza się automatycznie, gdy wandb jest zalogowany (`WANDB_API_KEY` albo wpis w `~/.netrc` po `wandb login`).
  Bez logowania trening idzie dalej z komunikatem „wandb: off”. Projekt domyślnie `trenowanie_generatora_obrazkow`
  (`--wandb <projekt>`), `--no-wandb` wyłącza.
- Nazwa runu = nazwa katalogu `--out`. ID runu jest w `<out>/wandb_id.txt`, więc wznowiony trening dopisuje się do tego
  samego runu w wandb.
- Co `--log-every` kroków: `train/loss_fm`, `train/loss_mlm`, `train/grad_norm`, `lr_mult`, `steps_per_s`, `imgs_per_s`,
  `tflops_per_gpu`, `mfu`. Co `--eval-every`: `val/loss` (EMA) i `val/loss` w 4 przedziałach t. Config = wszystkie argumenty.

## 3. Szacunki

- **Dysk** (maszyna treningowa): ~370 GB na przekonwertowane dane + ~10 GB captionów. `convert --download --delete-raw`
  pobiera i konwertuje shard po shardzie, więc surowe `.npz` nie leżą na dysku jednocześnie. Ten komputer (73 GB wolne) się nie nadaje.
- **RAM**: im więcej, tym lepiej dla page cache, ale nie jest to wymagane. Workerzy nie ładują danych do RAM, wszystko idzie przez mmap.
- **VRAM** (pomiar na 3090 Ti, 116M): ~10 GB przy mb 32, ~11 GB przy mb 64. Mb 256 szacunkowo ~17–18 GB, do sprawdzenia przy starcie.
- **Czas (116M, AdamW)**: RTX 5080 daje ~383 img/s na kartę przy mb 32; 3090 Ti ma podobny throughput bf16 z akumulacją fp32.
  Przy skalowaniu ~80% wychodzi ~2.5k img/s na 8 kart, czyli **~6–7 h** na 63.4M próbek (247 805 kroków × 256).
  Z Muonem ~25% dłużej. Dla `50m` ~3.5–5 h. Prawdziwą liczbę pokaże pierwsze 1000 kroków (log `img/s`, `eta`).

## 4. Kroki

```bash
# osobny venv, szczegóły w README
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128 && uv pip install -r requirements.txt
# referencyjny model: tokenizer, dekoder FLUX.2, pipeline.py, wagi do porównania
huggingface-cli download AuroraAI-Research/AuroraIMG2-50M-Preview --local-dir AuroraIMG2-50M-Preview

# 0. sprawdzić format captionów w prawdziwym shardzie (parser jest odporny na listy/dict/JSON, ale trzeba to zobaczyć)
python prepare_data.py convert --out data/aurora --download --limit 1
python prepare_data.py inspect data/aurora/_raw/data/<shard>.npz   # (albo pobrać ręcznie 1 shard bez --delete-raw)

# 1. dane: domyślnie 10% (~39 GB); później `--fraction 1` do tego samego katalogu dociąga resztę (~385 GB razem)
python prepare_data.py convert  --out data/aurora --download --delete-raw --workers 6
python prepare_data.py tokenize --data data/aurora --tokenizer AuroraIMG2-50M-Preview/tokenizer/tokenizer.json

# 2. punkt odniesienia: val loss opublikowanych wag na naszym hold-oucie (to jest liczba do pobicia)
python train.py --data data/aurora --out runs/ref --eval-only --load-hf AuroraIMG2-50M-Preview

# 3. krótki run kontrolny na 8 GPU (~300 kroków przy batchu 2048): sprawdzić img/s, czy loss spada, czy nie ma NaN
torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/try --size 116m --steps 300 --eval-every 100

# 3b. sweep lr dla macierzy ukrytych (wartości przy batchu 256, skalowane x2.83): ~600 kroków każdy, najniższy EMA val/loss
for lr in 5e-5 1e-4 2e-4 4e-4; do
  torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/lr_$lr --size 116m --steps 600 \
    --eval-every 100 --lr-hidden $lr --merge-every 0 --keep-every 0
done
# 3c. (opcjonalnie) A/B z Muonem; porównać val/loss względem czasu zegarowego, nie kroków
torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/ab_muon --size 116m --optimizer muon

# 4. trening na 10% danych (domyślnie: batch 2048, 3 098 kroków, ~11 epok; wznawia się sam z runs/p10/latest.pt)
torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/p10 --size 116m --lr-hidden <z 3b>

# 4b. jeśli pomysł działa: całość (najpierw convert --fraction 1 + tokenize), 247 805 kroków
torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/aurora --size 116m --lr-hidden <z 3b> --fraction 1

# 5. WSM: uśrednienie checkpointów z końcówki (wybór schematu i okna na val), eksport do formatu HF + siatka 8 promptów
python merge.py runs/p10                       # -> runs/p10/merged.pt, tabela val/loss: surowe / EMA / WSM
python export.py runs/p10/merged.pt exports/p10   # (albo runs/p10/latest.pt dla EMA)
python sample.py exports/p10 --out grid.png
python sample.py AuroraIMG2-50M-Preview --out grid_ref.png
```

Najpierw `torchrun --nproc_per_node 8 tests/check_nccl.py`: test samej komunikacji, z przepustowością i szacunkiem kosztu all-reduce
na krok. Na `adamo-rigga` (zmodowany sterownik z P2P) trzeba **`NCCL_P2P_LEVEL=SYS`**. Bez tego NCCL nie używa P2P między kartami
pod różnymi root portami, próbuje `/dev/shm` i pada z `Error while attaching to shared memory segment`. Szczegóły w README
(sekcja „NCCL na kartach GeForce”). Z P2P komunikacja jest szybsza, niż zakładałem w szacunkach (tam było PCIe przez host),
więc narzut DDP i all_gather Muona powinien być mniejszy.

## 5. Kryteria sukcesu
1. `val/loss` EMA na końcu ≤ wartość z kroku 2 (opublikowane wagi 50M). Model 116M powinien ją pobić. To główna, tania metryka.
   Rozbicie na koszyki `t` pokazuje, gdzie ewentualnie odstajemy.
2. Siatka 8 promptów wizualnie porównywalna z `samples/grid.png`.
3. (opcjonalnie) GenEval 2: ~6.0 GM / ~36 AM przy 50 krokach Eulera, CFG 3.0.

## 6. Pliki
| plik | co robi |
|---|---|
| `aurora/model.py` | text encoder + DiT, nazwy wag 1:1 z HF, presety `SIZES` (`50m`, `116m`) |
| `aurora/muon.py` | Muon: batch po kształtach + podział między ranki + all_gather |
| `aurora/data.py` | mmap store, deterministyczny i wznawialny strumień, hold-out |
| `prepare_data.py` | `inspect` / `convert` (pobieranie + konwersja) / `tokenize` |
| `train.py` | trening DDP, Muon+AdamW, EMA, MLM, eval, checkpointy, wandb |
| `merge.py` | WSM: ważone uśrednianie checkpointów z `merge/`, wybór schematu i okna na walidacji |
| `export.py` | checkpoint → katalog w formacie HF (działa z oryginalnym `pipeline.py`) |
| `sample.py` | siatka promptów z README (wymaga diffusers z `AutoencoderKLFlux2`) |
| `tests/check_model.py` | zgodność z opublikowanymi wagami i referencyjnym kodem |
| `tests/check_muon.py` | Muon dzielony == zwykły Muon, wagi identyczne na rankach (1 proces i `torchrun` 2×) |

## 7. Status testów (lokalnie, 1× RTX 5080)
- ✅ architektura: strict load + identyczny output co referencja, 12.59M + 50.43M parametrów
- ✅ pipeline danych na syntetycznych shardach w formacie datasetu (list/dict captiony, obrazy bez captionów)
- ✅ trening 1 GPU z `torch.compile` + akumulacja, wznowienie z checkpointu
- ✅ ścieżka DDP (2 procesy, gloo, hook bf16, `no_sync`)
- ✅ eksport → eval-only daje ten sam val loss co EMA w treningu
- ✅ Muon dzielony: zgodny ze zwykłym w granicach szumu bf16, identyczne wagi na 2 rankach
- ✅ `--size 116m`: trening 1 GPU (Muon i AdamW), DDP 2 ranki z dzielonym Muonem, wznowienie, eksport z configiem 768×12
- ✅ czysty venv (uv, torch 2.11.0+cu128, diffusers 0.42.1): testy, trening, eksport, `sample.py` na opublikowanych wagach
  (siatka zgodna z kartą modelu) i na wyeksportowanym 116M
- ⏳ nie testowane: prawdziwe shardy, NCCL na 8 kartach

## Źródła
- Surjo-Image-Preview, karta modelu (przepis AdamW, batch 1024, lr 2e-4)
- [Malladi i in. 2022, On the SDEs and Scaling Rules for Adaptive Gradient Algorithms](https://arxiv.org/abs/2205.10287): reguła √ dla Adama
- [Zheng i in. 2025, Scaling Diffusion Transformers Efficiently via μP](https://arxiv.org/abs/2505.15270v2): μP dla DiT/PixArt/MMDiT
- [Busbridge i in. 2023, How to Scale Your EMA](https://arxiv.org/pdf/2307.13813): skalowanie EMA z batchem
- Peebles & Xie 2023, DiT: AdamW 1e-4, batch 256, EMA 0.9999
