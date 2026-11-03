#!/usr/bin/env bash
# Start (or resume) the Polish caption translation (translate_captions.py): tmux session "tlum", one window per GPU
# with an automatic restart after a crash, plus a "postep" window counting finished shards. Finished shards are
# skipped, so running this again after a crash or a reboot continues where it stopped (only the shard in progress
# is redone).
#   bash translate_tmux.sh && tmux attach -t tlum
#   crontab -e:  @reboot cd /path/to/trenowanie_generatora_obrazkow && bash translate_tmux.sh
cd "$(dirname "$0")"
DATA=${DATA:-data/aurora_full}
if tmux has-session -t tlum 2>/dev/null; then
  echo "tmux session tlum is already running (tmux attach -t tlum)"
  exit 0
fi
tmux new-session -d -s tlum -n postep -c "$PWD" \
  "watch -n 60 'echo \$(ls $DATA/*.caps_pl.json 2>/dev/null | wc -l) / \$(ls $DATA/*.caps.json | wc -l) shards; date'"
for i in 0 1 2 3 4 5 6 7; do
  tmux new-window -t tlum -n gpu$i -c "$PWD" \
    "ulimit -n 100000; until CUDA_VISIBLE_DEVICES=$i .venv-vllm/bin/python translate_captions.py --data $DATA --rank $i --world 8; do echo restart in 30 s; sleep 30; done; read"
done
echo "started: tmux attach -t tlum   (Ctrl-b 0 = progress, Ctrl-b 1..8 = GPU 0..7, Ctrl-b d = detach)"
