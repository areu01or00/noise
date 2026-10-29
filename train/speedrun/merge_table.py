"""Re-choose run.py's MERGE_TABLE (Slayer Merge): per-tensor coefficients a in w + a (avg - w), on one GPU.

1. Save a run's weights and averages just before the merge (timed, so not a record run):
       uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py --save-premerge premerge.pt
2. Sweep each tensor's coefficient on identity-phase training rows (largest tensors first), check on disjoint ones:
       uv run train/speedrun/merge_table.py premerge.pt
   It prints MERGE_TABLE to paste into run.py. No validation rows are read.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import utils.model as M  # noqa: E402
from utils.prepare_data import TRAIN_ROWS, load_rows  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('premerge', type=Path)
p.add_argument('--data-dir', type=Path, default=Path('data'))
p.add_argument('--identity-rows', type=int, default=355 * 8, help='run.py IDENTITY_UPDATES x BATCH')
p.add_argument('--rows', type=int, default=256, help='rows to pick on, and as many to check on')
p.add_argument('--sweeps', type=int, default=2)
p.add_argument('--alphas', default='-0.25,0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0,1.125,1.25')
p.add_argument('--decay', type=float, default=0.02, help='which average to pull toward')
p.add_argument('--start', type=Path, help='json {name: a} to start from (default 0.75 everywhere, #52)')
p.add_argument('--out', type=Path, default=Path('merge_table.json'))
args = p.parse_args()
ALPHAS = tuple(float(a) for a in args.alphas.split(','))
dev, SEED = 'cuda', 0

# identity-phase rows, as run.py draws them: trained on before every Cut update, so the end weights treat them
# much like unseen rows; split into disjoint pick and check sets
train = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS).long()
ident = train[torch.from_numpy(np.random.default_rng((SEED + 1, 0)).permutation(len(train))[:args.identity_rows])]
order = torch.randperm(len(ident), generator=torch.Generator().manual_seed(7))
pick = ident[order[:args.rows]].to(dev)
check = ident[order[args.rows:2 * args.rows]].to(dev)

saved = torch.load(args.premerge, map_location='cpu', weights_only=True)
state = {n: v.to(dev) for n, v in saved['state'].items()}
avg = {n: v.to(dev).float() for n, v in saved['means'][list(saved['decays']).index(args.decay)].items()}
with torch.device('meta'):
    model = M.GPT(M.GPTConfig())
model.to_empty(device=dev)
model.init_weights()
model.load_state_dict(state, strict=True)
model.eval()
live = model.state_dict()


@torch.no_grad()
def loss(rows):
    total = 0.0
    for b in rows.split(32):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total += model(b[:, :-1].contiguous(), b[:, 1:].contiguous(), loss_reduction='sum').double().item()
    return total / (len(rows) * (rows.shape[1] - 1))


def put(n, a):
    live[n].copy_((state[n].float() + a * (avg[n] - state[n].float())).to(state[n].dtype))


for n in avg:
    put(n, 0.75)
ref_pick, ref_check = loss(pick), loss(check)
print(f'0.75 everywhere: pick {ref_pick:.5f}, check {ref_check:.5f} ({len(avg)} tensors)', flush=True)
table = json.loads(args.start.read_text()) if args.start else {n: 0.75 for n in avg}
for n in avg:
    put(n, table[n])
cur = loss(pick)
for sweep in range(args.sweeps):
    for n in sorted(avg, key=lambda k: -avg[k].numel()):
        best = (cur, table[n])
        for a in ALPHAS:
            if a != table[n]:
                put(n, a)
                trial = loss(pick)
                if trial < best[0]:
                    best = (trial, a)
        cur, table[n] = best
        put(n, table[n])
    print(f'sweep {sweep}: pick {cur - ref_pick:+.5f}, check {loss(check) - ref_check:+.5f}', flush=True)
args.out.write_text(json.dumps(table, indent=1))
print('MERGE_TABLE = {\n' + ''.join(f'    {n!r}: {a},\n' for n, a in table.items()) + '}')
