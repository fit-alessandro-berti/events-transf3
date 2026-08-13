# Matched training-debug comparison

Baseline: `baseline`.

The joint score equally weights held-out classification NLL and raw-hour regression MAE after normalization by baseline epoch 1. Lower is better.

| Run | Epochs | Best class NLL | Best class accuracy | Best regression MAE | Best joint score | Last confidence gap | Mean clip fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 20 | 1.90837 (e20) | 0.305672 (e16) | 837.32 (e13) | 0.806643 (e16) | 0.165994 | 0.997333 |
| clip5 | 20 | 1.87592 (e20) | 0.310108 (e15) | 841.872 (e6) | 0.803059 (e14) | 0.173071 | 0.553167 |
| clip10 | 20 | 1.86053 (e20) | 0.304426 (e16) | 834.176 (e6) | 0.805314 (e15) | 0.174891 | 0.0716667 |
| smoothing010 | 20 | 1.92285 (e19) | 0.299489 (e20) | 845.103 (e14) | 0.811981 (e20) | 0.160432 | 0.997167 |
| regression_balanced | 20 | 1.90843 (e20) | 0.305718 (e19) | 840.106 (e6) | 0.811616 (e16) | 0.161095 | 0.991167 |
| head_focused | 20 | 1.7683 (e20) | 0.328596 (e17) | 837.706 (e13) | 0.786409 (e17) | 0.184606 | 0.983667 |
