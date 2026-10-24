#!/usr/bin/env bash
# Reproduces every number, table and figure in the reports, in order.
# CPU only. Roughly 6-8 hours on a 14-core laptop, most of it Part 1 training.
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
#   bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

python download_data.py                         # both Kaggle datasets

# ---- Part 1
python audit.py            | tee logs/audit.log  # inventory, pixel stats, duplicates, val-set CI
python split.py            | tee logs/split.log  # patient-grouped split, locked test, 5 folds, image cache
python train.py --exp shortcut   | tee logs/shortcut.log
python train.py --exp baselines  | tee logs/baselines.log
python train.py --exp leak       | tee logs/leak.log
python train.py --exp transfer   | tee logs/transfer.log
python train.py --exp arogya     | tee logs/arogya.log
python train.py --exp twobranch  | tee logs/twobranch.log
python train.py --exp res_aug    | tee logs/res_aug.log

# ---- Part 2 (validation only until evaluate_test.py)
python part2_tune.py       | tee logs/tune.log
python final_model.py      | tee logs/final_model.log
python part2_threshold.py  | tee logs/threshold.log
python part2_cv.py         | tee logs/cv.log
python evaluate_test.py    | tee logs/test.log      # the only reader of the locked test set
python external_eval.py    | tee logs/external.log  # the only reader of the external hospital data
python ship_check.py       | tee logs/ship.log

python make_tables.py                            # results/*.json -> reports/tables.md
echo "done. demo: python app.py"
