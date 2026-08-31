#!/usr/bin/env bash
# End-to-end smoke test: every entry point, every coreset selector, both ICD formulations.
#
#     bash scripts/smoke_test.sh [output_dir]
#
# Runs each path at a deliberately tiny scale -- a handful of iterations, one evaluation
# network trained for two epochs -- so the whole thing finishes in a few minutes on one GPU.
# It checks that the code runs, not that it is accurate; the accuracies it prints are
# meaningless. Exits non-zero on the first failure.
set -euo pipefail

OUT="${1:-runs/smoke}"
CFG_SMALL="data.ipc=2 eval.num_eval=1 eval.epochs=2"
mkdir -p "$OUT"

echo "=== configs load and validate ==="
python - <<'PY'
import glob, sys
from ddm.config import load_config
bad = 0
for f in sorted(glob.glob('configs/**/*.yaml', recursive=True)):
    if f == 'configs/base.yaml' or '/_base' in f:
        continue
    try:
        load_config(f)
    except Exception as e:
        print(f'FAIL {f}: {e}'); bad += 1
print(f'{bad} config failures')
sys.exit(1 if bad else 0)
PY

echo
echo "=== condensation: released method (bounded ICD) ==="
python train.py -c configs/ours/cifar10_ipc10.yaml \
    --set $CFG_SMALL condense.iterations=6 condense.num_exp=1 eval.every=3 \
          output.save_path="$OUT/ours"

echo
echo "=== condensation: published objective (kl ICD, legacy accumulator) ==="
python train.py -c configs/paper/legacy_cifar10_ipc10.yaml \
    --set $CFG_SMALL condense.iterations=4 condense.num_exp=1 \
          output.save_path="$OUT/paper"

echo
echo "=== condensation: plain distribution matching ==="
python train.py -c configs/dm/cifar10_ipc10.yaml \
    --set $CFG_SMALL condense.iterations=4 condense.num_exp=1 \
          output.save_path="$OUT/dm"

echo
for s in random herding kcenter kmeans forgetting uncertainty el2n grand; do
    echo "=== coreset: $s ==="
    python train.py -c "configs/coreset/cifar10_ipc10_$s.yaml" \
        --set $CFG_SMALL coreset.num_exp=1 coreset.proxy.epochs=1 \
              coreset.proxy.num_models=1 output.save_path="$OUT/$s"
done

echo
echo "=== evaluate: cross-architecture, from a checkpoint ==="
python evaluate.py -c configs/eval/cross_arch.yaml \
    -k "$OUT/ours/condensed_CIFAR10_ConvNet_style_2ipc.pt" \
    --set eval.num_eval=1 eval.epochs=2 output.save_path="$OUT/cross_arch"

echo
echo "=== collect results ==="
python scripts/collect_results.py "$OUT"

echo
echo "SMOKE TEST PASSED"
