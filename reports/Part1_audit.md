# Part 1 - Audit of Rohit's data, then the models

Mitu Kumar (B24CS029)

Every number below comes from a json in `results/`, produced by the script named next to it.
`reports/tables.md` (made by `make_tables.py`) has all the tables in one place.
Training was run on a Kaggle T4 GPU with `run_all.sh`; the exact split that run used is saved in
`results/split_used.csv` (file name -> test / val / train).

---

## 1. Inventory and pixel statistics (`audit.py`)

| folder | NORMAL | PNEUMONIA |
|---|---|---|
| train | 1341 | 3875 |
| test | 234 | 390 |
| val | 8 | 8 |

5,856 files. 5,573 are stored as 8-bit greyscale and 283 as RGB, but all 283 RGB files have three identical
channels, so all 5,856 are true greyscale.

| statistic | NORMAL median | PNEUMONIA median | ROC-AUC of this number alone |
|---|---|---|---|
| width (px) | 1654 | 1160 | 0.889 |
| height (px) | 1323 | 776 | **0.915** |
| aspect w/h | 1.23 | 1.49 | 0.840 |
| mean brightness | 122.4 | 122.9 | 0.508 |
| contrast (std) | 61.5 | 54.8 | 0.698 |

Figure: `figures/p1_size_brightness_hist.png`.

Mean brightness is the same for both classes. **Image size is not**: normal films are about 2.4 MP,
pneumonia films about 0.9 MP and wider. Height on its own separates the classes with AUC 0.915.
This is a property of how the images were acquired and saved, not of the lungs. A model trained on these
folders can score well by learning "small, wide file = pneumonia", and that rule will not exist at a camp,
where every child is imaged on the same machine. Resizing to a square does not remove it either, because a
wide image squashed to a square has visibly distorted anatomy.

## 2. Duplicates and leakage (`audit.py`)

- **Exact duplicates (md5):** 30 groups, 62 files, all inside one folder (22+2 in train, 6 in test).
- **pHash alone is misleading here.** At <= 4 bits, pHash flags 265 pairs, 26 of them between train and test.
  When I looked at them they were different children: paediatric AP films all look alike at an 8x8 DCT. So I used
  pHash (<= 10 bits) only to find candidates, and confirmed each one with pixel correlation on a 64x64 version
  (NCC >= 0.97).
- **Confirmed near duplicates:** 34 pairs, 28 within train and 6 within test, usually the same patient's file saved twice.
  **Across train and test: zero.** The most similar train/test pair has NCC 0.965 and is two different patients.
  `figures/p1_worst_pairs.png` (and the wider `figures/p1_worst_pairs_wide.png`) shows the five closest train/test pairs side by side, and `figures/p1_true_duplicate.png` shows a real duplicate for comparison.
- **Patient ids.** The same `personNNN` string appears in both train and test (170 ids). I checked whether
  these are the same children: pairs with the same id inside train are slightly more alike than random pairs
  (median pHash distance 24 vs 26), but across train and test same-id pairs are no closer than random (24 vs 24).
  So the test folder numbers its patients separately, and these ids are not evidence of leakage. I did not report them as such.

**Conclusion.** Rohit's original train/test folders do not leak duplicates into each other. The data still has
real problems, though: the acquisition shortcut above, 1,158 patients with more than one image (up to 30), and a
16-image validation folder.

## 3. The validation set

16 out of 16 correct gives a 95% confidence interval for the true accuracy of **[0.794, 1.000]** (Clopper-Pearson,
lower bound = 0.025^(1/16)); Wilson gives [0.806, 1.000]. "100% on val" is compatible with a model that is wrong
one time in five. It cannot be used to choose epochs, thresholds or models.

**Replacement (`split.py`).** I pooled all 5,856 images, put images of the same patient (per folder) and confirmed
near duplicates into one group (3,287 groups), and then:

- held out a **locked test set** of ~1/7 of the groups (821 images: 227 normal, 594 pneumonia), stratified by
  class with `StratifiedGroupKFold`. It is read by exactly one script, `evaluate_test.py`, after everything was frozen.
- split the remaining pool (5,035 images) into **5 stratified, patient-grouped folds**. Fold 0 (960 images) is
  the validation set for all Part 1 experiments and for threshold choice. All five folds are used for the CV in Part 2.

Why both, and not k-fold alone: if I used k-fold alone, every number I tuned (architecture, epochs,
threshold, refer band) would be chosen on the same data I report, and the estimate would be optimistic. A locked
test set alone, with a single validation split, has no measure of spread. Using a locked test plus CV on the
pool gives an untouched final number and a mean ± sd. The cost is ~15% fewer training images, which I accepted.

## 4. Shortcut hunt (`train.py --exp shortcut`)

Logistic regression, evaluated on fold-0 validation (chance = 0.731):

| crippled input | accuracy | bal. acc | ROC-AUC |
|---|---|---|---|
| whole image at 16x16 pixels | 0.941 | 0.923 | 0.980 |
| **16x16 with the lung field greyed out** | **0.894** | 0.853 | **0.941** |
| 3 numbers only: mean intensity, aspect, area | 0.845 | 0.777 | 0.912 |
| 2 numbers only: aspect, area | 0.842 | 0.778 | 0.905 |

With the lungs removed, 256 pixels of border, shoulders and abdomen still give AUC 0.94. Two numbers about
the file size give AUC 0.91. So most of the "accuracy" on this dataset is available without looking at the lungs at all.
Any model trained here, including Rohit's, can use these cues, and nothing in a same-hospital test set would reveal it.

## 5. Baselines (fold-0 validation, threshold 0.5)

| model | acc | bal. acc | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| majority class (always pneumonia) | 0.731 | 0.500 | 0.731 | 1.000 | 0.845 | 0.731 |
| HOG (128px, 9 orient., 16px cells) + logistic regression | 0.963 | 0.949 | 0.970 | 0.979 | 0.974 | 0.996 |

HOG + LR (ROC-AUC 0.990) is the bar a CNN has to clear, and it is a high bar.

## 6. Transfer baseline: ResNet18 (3 seeds)

**Expectation, before running:** ImageNet filters are learned on colour photos, but the low-level ones (edges,
blobs, textures) should transfer to greyscale X-rays; the high-level ones (dog faces, wheels) should not matter much.
I expected fine-tuning to beat both HOG and a small net trained from scratch, by a little.

| | acc | bal. acc | recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| ResNet18, 224px, fine-tuned 8 epochs | 0.980 ± 0.002 | 0.971 ± 0.004 | 0.989 ± 0.002 | 0.996 ± 0.000 | 0.999 ± 0.000 |

The numbers agree: it is the best single model (about +0.006 AUC over HOG and ArogyaNet) and the most stable across
seeds. But it is 11.2M parameters and 44.8 MB as a file, which already breaks the 25 MB update channel.

## 7. ArogyaNet

Own design, no pretrained weights, **417,681 parameters**. Each block is
depthwise-separable 3x3 conv -> depthwise-separable 3x3 conv -> squeeze-and-excitation -> + skip -> ReLU,
with 4 stages (32/64/128/256 channels) and global average pooling instead of flatten+dense.
Input is 1x128x128.

**(a) Architecture table** (`results/arogyanet_table.md`), diagram in `figures/p1_arogyanet_diagram.png`:

| layer | output shape | params |
|---|---|---|
| stem conv3x3 s2 | 24 x 64 x 64 | 264 |
| stage1 Block s2 / s1 | 32 x 32 x 32 | 3,920 / 3,432 |
| stage2 Block s2 / s1 | 64 x 16 x 16 | 10,728 / 10,952 |
| stage3 Block s2 / s1 | 128 x 8 x 8 | 39,888 / 40,336 |
| stage4 Block s2 / s1 | 256 x 4 x 4 | 153,504 / 154,400 |
| global avg pool | 256 | 0 |
| dropout + linear | 1 | 257 |
| **total** | | **417,681** |

**(b) Hypotheses** (written before the ablation runs, see `reports/hypothesis_arogyanet.md`):

1. Residual skips give the gradient a short path through 8 blocks; expected +0.5 to 1 AUC point and a smaller seed spread.
2. SE re-weights channels with global context; expected +0.3 to 0.5 AUC points, possibly inside the noise.

**(c) Ablations** (3 seeds each, paired by seed against the full model, fold-0 validation):

| variant | params | ROC-AUC | Δ AUC vs full | Δ bal. acc |
|---|---|---|---|---|
| full ArogyaNet | 417,681 | 0.990 ± 0.001 | - | - |
| no residual | 372,945 | 0.978 ± 0.003 | **-0.0118 ± 0.0029** | -0.023 ± 0.021 |
| no SE | 372,561 | 0.986 ± 0.004 | -0.0042 ± 0.0033 | -0.013 ± 0.017 |

- **Residual: hypothesis supported, and the effect is bigger than I guessed.** -1.2 AUC points, four times its spread, and
  the no-skip version also drops recall from 0.978 to 0.945. The smaller-spread part did not hold (0.003 vs 0.001).
- **SE: inside the noise.** Removing it lowered AUC on all three seeds (by 0.0006, 0.0049, 0.0071), so the direction matches
  my hypothesis, but the mean (-0.0042) is smaller than two standard deviations of the difference (2 x 0.0033) and one seed
  barely moved. With 3 seeds that is not enough to claim SE helps; it may, by a fraction of a point. I kept it because it
  costs 45k parameters and does not hurt.

ArogyaNet (AUC 0.990) ties HOG + LR and is below ResNet18 (0.996).

## 8. Two-branch ArogyaNet (`--exp twobranch`)

Image branch + a small dense branch for (mean intensity, aspect ratio, area), concatenated before the classifier.

| | ROC-AUC | bal. acc |
|---|---|---|
| image only | 0.990 ± 0.001 | 0.941 ± 0.011 |
| image + metadata | 0.991 ± 0.001 | 0.951 ± 0.009 |
| paired difference | +0.0008 ± 0.0014 | +0.010 ± 0.011 |

The metadata branch adds nothing measurable; the difference is inside the noise. Connecting this to task 4:
the three numbers alone give AUC 0.91, so they clearly carry the label. They add nothing here because the image
branch already has that information: the shortcut model in task 4 got AUC 0.94 from a 16x16 border with no lungs.
If the branch had helped, it would mean the model had learned "which room / which scanner settings", not "what pneumonia looks like".
At a camp every child is imaged on one machine with one protocol, so file size and aspect carry no information about
the disease and that part of the model would be useless or harmful. I would not ship the two-branch model.

## 9. Two ablations: resolution and augmentation (`--exp res_aug`)

| variant | ROC-AUC | bal. acc | Δ AUC vs full (128px, rot+zoom) |
|---|---|---|---|
| 96 px | 0.991 ± 0.001 | 0.952 ± 0.006 | +0.0004 ± 0.0006 (noise) |
| 224 px | 0.991 ± 0.000 | 0.954 ± 0.003 | +0.0008 ± 0.0011 (noise) |
| no augmentation | 0.986 ± 0.002 | 0.941 ± 0.002 | -0.0040 ± 0.0009 |
| flips + rotation + zoom | 0.990 ± 0.001 | 0.944 ± 0.000 | +0.0001 ± 0.0018 (noise) |

- Resolution 96 vs 224 makes no difference (both within 0.001 AUC of 128). Diffuse pneumonia opacity is a
  low-frequency pattern, and the shortcut cues are large-scale too. 224 costs ~3.5x the compute for nothing, so I stayed at 128.
- Augmentation helps a little (-0.4 AUC points without it, outside the noise).
- **The medically wrong augmentation is the horizontal flip.** A flipped chest film shows the heart on the right
  (dextrocardia / situs inversus), the stomach bubble on the wrong side and the "R" marker mirrored. Any finding that depends
  on side (right-lower-lobe vs left) changes meaning, so the label of a flipped image is no longer the label of a real patient.
  It also teaches the model that a mirrored anatomy is normal, which in reality is a finding. Here it made no measurable
  difference to the score, but that only means this score cannot see the problem. The final model uses rotation (±10°),
  zoom (0.9-1.1) and small shifts, with **no flips**. (Vertical flips are worse still and nobody should use them.)

## What did Rohit do wrong? (summary for Part 2)

1. He trusted a 16-image validation folder (CI 79-100%).
2. He reported accuracy on a test set from the **same hospital and the same acquisition pipeline** as training, where file size alone gives 84% accuracy.
3. I tested the other common explanation for a 98% headline, an image-level random split that lets the same child into
   train and test (`train.py --exp leak`). On this data it makes **no difference**: HOG/LR 0.961 vs 0.961, ArogyaNet 0.940 vs 0.952.
   So I do not think his number was faked or leaked. ResNet18 gets 0.980 on my clean split, so 98.4% is a believable
   in-distribution number. **The problem is that it measured the wrong thing**: Part 2 shows what happens on another hospital's data.
4. ResNet50 is ~98 MB, so it never fit the 25 MB channel.
