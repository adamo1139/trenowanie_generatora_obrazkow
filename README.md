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

## Text encoder: Slayer149 (zamrożony) + REPA z PE-Spatial-B16 (domyślnie)

- `--te-size slayer149` (domyślnie): zamrożony [SlayerLab/Slayer149](https://huggingface.co/SlayerLab/Slayer149) (149M,
  angielski) + trenowany 2-blokowy refiner (~12M). Wymaga osobnej tokenizacji captionów (obok istniejącej):
  `python prepare_data.py tokenize --data data/aurora_full --encoder slayer149`.
  `--te-size 13m` / `27m` to dawne enkodery trenowane od zera (z MLM).
- **REPA** (`--repa-weight 0.5`, domyślnie): stan DiT po ~30% głębokości jest dopasowywany (cosinus, przez MLP) do cech
  [PE-Spatial-B16](https://huggingface.co/facebook/PE-Spatial-B16-512) (Apache-2.0) z obrazka zdekodowanego z latentu.
  Dekoder jest drogi, więc REPA liczy się tylko dla 25% każdego micro-batcha (`--repa-frac`) i przez pierwsze 40%
  treningu (`--repa-until`). `--repa-weight 0` wyłącza.
- Eksport modeli z enkoderem Slayer149 do formatu diffusers nie jest jeszcze zaimplementowany.

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
