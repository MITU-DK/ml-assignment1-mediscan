# MediScan audit - ML practical assignment

Mitu Kumar (B24CS029). PyTorch, CPU or GPU.

## Run everything
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
bash run_all.sh            # downloads data, runs every experiment, writes results/ figures/ reports/tables.md
```
On a Kaggle GPU (internet on) the whole thing is: `git clone ...`, `pip install imagehash`, `bash run_all.sh`.

## Use the model
```
python predict.py some_xray.jpeg     # class, probability, refer / reject decision
python app.py                        # Gradio demo at http://127.0.0.1:7860
```

## Files
| file | what it does |
|---|---|
| download_data.py | gets both Kaggle datasets (or links them if added as Kaggle inputs) |
| audit.py | Part 1 tasks 1-3: inventory, pixel stats, duplicates (pHash + NCC), val-set CI |
| split.py | patient/duplicate-grouped split: locked test + 5 folds, image cache |
| train.py | Part 1 tasks 4-9 (`--exp shortcut / baselines / leak / transfer / arogya / twobranch / res_aug`) |
| models.py, engine.py, common.py | ArogyaNet + ResNet18, training loop, metrics |
| part2_tune.py | random search + regularisation curves |
| final_model.py | final model, class-weighted and CLAHE variants, OOD guard |
| part2_threshold.py | class weights vs threshold moving, cost-based threshold, refer band |
| part2_cv.py | 5-fold grouped CV |
| evaluate_test.py | the only reader of the locked test set: final metrics, ensembles, PCA/t-SNE |
| external_eval.py | the only reader of the external hospital data, run once |
| predict.py, app.py, ship_check.py | shipping: CLI, demo, size/latency/dog/blank checks |
| make_tables.py | results/*.json -> reports/tables.md |

## Reports
- `reports/technical_report.pdf` - the 5-page technical report
- `reports/Part1_audit.md` - Part 1 in detail
- `reports/CEO_report.md` - report to the CEO
- `reports/tables.md` - every number, with the script that produced it

## Note on the refer band
`part2_threshold.py` was fixed after the Kaggle run: the band now goes from the 99.5%-sensitivity point to the
98%-specificity point, so the threshold t* falls inside it. It was re-run locally (validation only) and is
`[0.109, 0.899]`. `results/p2_test.json` still has the old band in its `refer_band` field. The band numbers
in the reports come from `make_tables.py`, which applies the new band to the saved test predictions
(`results/preds/test_final.npz`). The test set was not re-read.

## Note on GPU determinism
Seeds are set and logged, but GPU runs use `cudnn.benchmark`, which is not bit-for-bit deterministic.
Re-running gives numbers within the reported seed / fold spread, not identical digits. Example from this run:
the CV fold-0 model and the final model (same data, settings and seed) miss 28 vs 9 pneumonia cases on fold 0.

## Note on the split
The reported numbers were produced on Kaggle. Its preinstalled package versions shuffled the
groups into folds differently from `requirements.txt` versions, so a rerun with the pinned versions
gives a different (equally valid) split and slightly different numbers. The split actually used is
saved in `results/split_used.csv` (file -> test_locked / val_fold0 / train_folds1to4).
