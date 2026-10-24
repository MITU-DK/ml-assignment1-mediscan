# ArogyaNet - hypotheses (written before the ablation results existed)

Written: 2026-10-24 14:25 IST, while the first ArogyaNet run (full model, seed 0) was in epoch 2. No ablation run had started.

1. **Residual skip connections** - they give the gradient a short path through the 8 blocks, so I expect
   the full model to train faster and reach a slightly higher validation ROC-AUC than the no-skip version
   (maybe +0.5 to 1 point), with a smaller spread over seeds.
2. **Squeeze-and-excitation** - it lets the network re-weight channels by global image context, which I expect
   to help a little (+0.3 to 0.5 AUC points) by focusing on diffuse opacity patterns; honestly it may well be
   inside the seed noise for a network this small.
