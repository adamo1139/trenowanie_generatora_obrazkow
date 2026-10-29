# Trenowanie generatora obrazów

Trening od zera małego modelu text-to-image 256×256 według przepisu
[AuroraIMG2-50M-Preview](https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview): latentny flow matching
(rectified flow) z DiT-em, text encoderem trenowanym razem z nim i zamrożonym dekoderem FLUX.2. Celujemy w 8× RTX 3090 Ti.

- Architektura jest 1:1 z oryginałem: opublikowane wagi ładują się w trybie strict i dają identyczny output (`tests/check_model.py`).
- Rozmiary: `--size 50m` (oryginał, 63M z text encoderem) i `--size 116m` (DiT 768×12, domyślny punkt startowy w planie).
- Domyślnie AdamW (lr 1e-4 dla macierzy ukrytych, WD 0.1, stały lr po warmupie). Oryginalny przepis z Muonem: `--optimizer muon --schedule cosine`.
- Dane: [QLNI/FLUX-Reason-6M-flux2-latents](https://huggingface.co/datasets/QLNI/FLUX-Reason-6M-flux2-latents), ~5.6M gotowych latentów (~384 GB).

Pełny plan, uzasadnienie hiperparametrów, szacunki czasu i pamięci: **[PLAN.md](PLAN.md)**.

## Środowisko (osobny venv)

Trenujemy w osobnym venv w katalogu repo (`.venv/`, ignorowany przez gita), żeby nie mieszać pakietów z systemowym
Pythonem. Wymagany jest [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # 3090 Ti (sm86) i nowsze karty
uv pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), 'GPU')"
```

Bez uv: `python3.11 -m venv .venv && source .venv/bin/activate && pip install torch --index-url
https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt`. Wszystkie dalsze komendy zakładają aktywny venv
(albo wywołanie `.venv/bin/python` / `.venv/bin/torchrun`). Sprawdzone: torch 2.11.0+cu128, diffusers 0.42.1, Python 3.11.

## Szybki start

```bash
source .venv/bin/activate
huggingface-cli download AuroraAI-Research/AuroraIMG2-50M-Preview --local-dir AuroraIMG2-50M-Preview

# dane: pobranie + konwersja do mmap .npy + tokenizacja
python prepare_data.py convert  --out /data/aurora --download --delete-raw --workers 6
python prepare_data.py tokenize --data /data/aurora

# trening na 8 GPU (wznawia się sam z <out>/latest.pt)
torchrun --nproc_per_node 8 train.py --data /data/aurora --out runs/aurora --size 116m

# eksport wag EMA do formatu diffusers + siatka próbek
python export.py runs/aurora/latest.pt exports/aurora
python sample.py exports/aurora --out grid.png
```

## Struktura

| plik | co robi |
|---|---|
| `aurora/model.py` | text encoder + DiT (nazwy wag zgodne z HF), presety rozmiarów |
| `aurora/muon.py` | Muon dzielony między ranki DDP (opcjonalny) |
| `aurora/data.py` | dane przez mmap, deterministyczny i wznawialny strumień, hold-out |
| `prepare_data.py` | `inspect` / `convert` / `tokenize` |
| `train.py` | trening DDP: AdamW lub Muon, EMA, aux MLM, eval, checkpointy, wandb |
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
