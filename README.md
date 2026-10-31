# Trenowanie generatora obrazów

Trening od zera małego modelu text-to-image 256×256 według przepisu
[AuroraIMG2-50M-Preview](https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview): latentny flow matching
(rectified flow) z DiT-em, text encoderem trenowanym razem z nim i zamrożonym dekoderem FLUX.2. Celujemy w 8× RTX 3090 Ti.

- Architektura jest 1:1 z oryginałem: opublikowane wagi ładują się w trybie strict i dają identyczny output (`tests/check_model.py`).
- Rozmiary DiT: `--size 50m` (oryginał), `116m` (768×12), `253m` (1024×16, domyślny). Text encoder: `--te-size 13m`
  (oryginał, 384×6) albo `27m` (512×8, domyślny). Domyślnie razem ~272M (DiT 245M z szerszym cross-attention + TE 26.7M).
  Odtworzenie oryginału: `--size 50m --te-size 13m`.
- Domyślnie **Muon** (lr 1e-2, WD 0.01, jak w oryginale) dla macierzy ukrytych, AdamW dla reszty, stały lr po warmupie
  + **WSM** (uśrednianie checkpointów zamiast wygaszania lr, `merge.py`). AdamW dla wszystkiego: `--optimizer adamw`.
  Warianty Muona pod DiT: `--cmuon` ([CMuon](https://arxiv.org/abs/2608.02502): osobna ortogonalizacja bloków
  macierzy złączonych - q/k/v, k/v cross-attention, gate/up SwiGLU) i `--muon-period 3` ([periodic row-wise
  Muon](https://arxiv.org/abs/2608.20818): Newton-Schulz co 3. krok, pomiędzy znormalizowane wiersze momentum × 0.15).
- Dane: [QLNI/FLUX-Reason-6M-flux2-latents](https://huggingface.co/datasets/QLNI/FLUX-Reason-6M-flux2-latents), ~5.6M gotowych latentów (~385 GB).
  **Domyślnie pobieramy i trenujemy na 10%** (~39 GB), żeby najpierw sprawdzić pomysł; `--fraction 1` = całość.

Pełny plan, uzasadnienie hiperparametrów, szacunki czasu i pamięci: **[PLAN.md](PLAN.md)**.

## Środowisko (osobny venv)

Trenujemy w osobnym venv w katalogu repo (`.venv/`, ignorowany przez gita), żeby nie mieszać pakietów z systemowym
Pythonem. Wymagany jest [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128  # najpierw!
uv pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), 'GPU')"
```

Bez uv: `python3.11 -m venv .venv && source .venv/bin/activate && pip install torch --index-url
https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt`. Wszystkie dalsze komendy zakładają aktywny venv
(albo wywołanie `.venv/bin/python` / `.venv/bin/torchrun`). Sprawdzone: torch 2.11.0+cu128, diffusers 0.42.1, Python 3.11.

## Text encoder: Slayer149 (zamrożony) + REPA z DINOv2-B/14 (domyślnie)

- `--te-size slayer149` (domyślnie): zamrożony [SlayerLab/Slayer149](https://huggingface.co/SlayerLab/Slayer149) (149M,
  angielski) + trenowany refiner: domyślnie 2 bloki o połowie szerokości denoisera (2B: 1024, ~25M; 750m:
  1280, ~36M; w PP siedzi na stage'u 0, więc większy go spowalnia), `--te-dim` / `--te-depth` zmieniają rozmiar, `--te-dim 768 --te-depth 2` = dawny ~13M
  (potrzebne do wznowienia starszych runów). Wymaga osobnej tokenizacji captionów (obok istniejącej):
  `python prepare_data.py tokenize --data data/aurora_full --encoder slayer149`.
  `--te-size 13m` / `27m` to dawne enkodery trenowane od zera (z MLM).
- **REPA** (`--repa-weight 0.5`, domyślnie): stan DiT po ~30% głębokości jest dopasowywany (cosinus, przez MLP) do cech
  [DINOv2-B/14 z rejestrami](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) (Apache-2.0) z obrazka
  zdekodowanego z latentu (przeskalowanego do 224 px → siatka 16×16 = tokeny DiT). PE-Spatial-B16:
  `--repa-encoder vit_pe_spatial_base_patch16_512.fb`. `--irepa` = iREPA (projektor conv 3×3 + normalizacja cech po
  tokenach). Dekoder jest drogi, więc REPA liczy się tylko dla 25% każdego micro-batcha (`--repa-frac`), co N-ty krok
  (`--repa-every N`) i przez pierwsze 40% treningu (`--repa-until`). `--repa-weight 0` wyłącza.
- Eksport (`export.py`) pakuje użyte warstwy Slayer149 i jego tokenizer do repo, a `pipeline.py` (`aurora/hf_pipeline.py`)
  obsługuje oba enkodery; `tests/check_export.py` sprawdza zgodność z modułami treningowymi.

## Szybki start

```bash
source .venv/bin/activate
huggingface-cli download AuroraAI-Research/AuroraIMG2-50M-Preview --local-dir AuroraIMG2-50M-Preview

# dane: pobranie + konwersja do mmap .npy + tokenizacja. Domyślnie 10% datasetu (~39 GB, ~0.57M obrazków);
# całość (~385 GB): dodać --fraction 1 (już pobrane shardy są pomijane)
python prepare_data.py convert  --out data/aurora --download --delete-raw --workers 6
python prepare_data.py tokenize --data data/aurora

# trening na 8 GPU. Domyślnie model 253m, 10% danych, batch 2048 (8 GPU x 64 x akumulacja 4), 3 098 kroków (~11 epok);
# całość: --fraction 1 (30 976 kroków). OOM: --micro-batch 32 (akumulacja liczy się sama). Kroki, lr i EMA przeskalowują się z batchem względem oryginału (256).
# Wznawia się sam z <out>/latest.pt (co 1/50 runu); trwałe snap_<krok>.pt co 1/10 runu.
# wandb włącza się sam, jeśli jest zalogowany (`wandb login`); projekt trenowanie_generatora_obrazkow, --no-wandb wyłącza.
torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/aurora --size 116m

# WSM: zamiast wygaszać lr, uśredniamy checkpointy z końcówki treningu (schemat i okno wybierane na walidacji)
python merge.py runs/aurora                                  # -> runs/aurora/merged.pt
# eksport do formatu diffusers + siatka próbek
python export.py runs/aurora/merged.pt exports/aurora
python sample.py exports/aurora --out grid.png
```

## Duże modele: pipeline parallel (1F1B), `train_pp.py`

Bloki DiT dzielone na `--pp` kart (Schedule1F1B z `torch.distributed.pipelining`). Stage 0 ma też text encoder, projekcję
czasu i `patch_in`, ostatni stage `norm`, `patch_out` i stratę. Między kartami idą stany tokenów, modulacja czasu, tokeny
tekstu i maska (oraz ich gradienty), strata REPA liczy się w stage'u z wybranym blokiem, a cele REPA (dekoder + DINOv2) liczą wszystkie karty po równo
(micro-batch k na karcie k mod pp) i wysyłają do tego stage'u. Presety: `--size 1b / 1.5b / 2b`
(szerokość 2048, 15/23/31 bloków) oraz `--size 1.6b / 2.4b` (szerokość 2560, head dim 128, 15/23 bloki; najlepsze TFLOPS na 3090 Ti wg `tests/bench_shapes.py`). Balans: zamrożony LM liczą wszystkie karty po równo (micro-batch k na karcie
k mod pp, stany wysyłane do stage'u 0), a stage 0, który ma jeszcze trenowany refiner (~1 blok pracy, `--te-cost`),
dostaje o 1 blok mniej; głębokość 8k−1 sprawia, że przy pp 8/4/2 wychodzi równo (2B na 8 kartach: [3, 4, 4, …]). Checkpointy są zbierane na rank 0 w tym samym formacie co z `train.py`, więc `export.py`,
`merge.py` i `tests/check_export.py` działają bez zmian; stan optymalizatora per stage w `<out>/opt_stage<s>.pt`.

```bash
NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node 8 train_pp.py --data data/aurora_full --fraction 1 --size 1.5b \
  --pp 8 --batch 512 --microbatches 32 --out runs/1p5b
# OOM: --act-checkpoint selective (trzyma wyniki matmuli i attention, przelicza tylko elementwise: ~2x mniej
#      aktywacji, bez straty prędkości) albo --act-checkpoint (pełny: prawie zero aktywacji, ~25% wolniej);
#      2 potoki po 4 karty: --pp 4 (+ --diloco H)
```

`tests/check_pp.py` (CPU, gloo, `torchrun --nproc_per_node 1/2/4`) sprawdza, że straty i gradienty (z REPA) są identyczne
jak w niepodzielonym modelu.

## NCCL na kartach GeForce

Najpierw sprawdzić samą komunikację między kartami (kilka sekund, bez danych i modelu):

```bash
torchrun --nproc_per_node 8 tests/check_nccl.py
```

- **Sterownik z odblokowanym P2P** (np. łatka tinygrad na open-gpu-kernel-modules): NCCL domyślnie włącza P2P tylko dla kart
  blisko siebie w topologii PCIe. Gdy ścieżka idzie przez CPU, trzeba to wymusić:
  `NCCL_P2P_LEVEL=SYS torchrun ...`. Sprawdzenie: `python -c "import torch; print(torch.cuda.can_device_access_peer(0, 1))"`.
- **Błąd `Error while attaching to shared memory segment /dev/shm/nccl-...`**: NCCL próbuje transportu przez `/dev/shm`.
  Przy działającym P2P naprawia to `NCCL_P2P_LEVEL=SYS`. Bez P2P trzeba sprawdzić `df -h /dev/shm` (miejsce, uprawnienia
  `drwxrwxrwt`), a w ostateczności `NCCL_SHM_DISABLE=1` (wolniejszy transport przez sockety).
- `NCCL_DEBUG=INFO` pokazuje, którą drogę wybrało NCCL (`via P2P`, `via SHM`, `via NET`).
- Te same zmienne środowiskowe daje się przed `torchrun ... train.py`.

## Struktura

| plik | co robi |
|---|---|
| `aurora/model.py` | text encoder + DiT (nazwy wag zgodne z HF), presety rozmiarów |
| `aurora/muon.py` | Muon dzielony między ranki DDP (opcjonalny) |
| `aurora/data.py` | dane przez mmap, deterministyczny i wznawialny strumień, hold-out |
| `prepare_data.py` | `inspect` / `convert` / `tokenize` |
| `train.py` | trening DDP: AdamW lub Muon, EMA, aux MLM, eval, checkpointy, wandb |
| `merge.py` | WSM: ważone uśrednianie checkpointów z końcówki treningu |
| `export.py` | checkpoint → katalog diffusers (działa z oryginalnym `pipeline.py`) |
| `sample.py` | siatka 8 promptów z karty modelu |
| `tests/` | zgodność z opublikowanymi wagami, poprawność Muona |

```bash
python tests/check_model.py
python tests/check_muon.py
DIST_BACKEND=gloo torchrun --nproc_per_node 2 tests/check_muon.py   # 2 ranki na jednej karcie
```

## Podziękowania i licencje

- Bloki modelu w `aurora/model.py` pochodzą z `pipeline.py` modelu
  [AuroraIMG2-50M-Preview](https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview) (Apache-2.0).
- Dekoder [FLUX.2 small decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) © Black Forest Labs (Apache-2.0).
- Muon: [Keller Jordan](https://kellerjordan.github.io/posts/muon/).
- Dane mają własne licencje, patrz karty datasetów.
