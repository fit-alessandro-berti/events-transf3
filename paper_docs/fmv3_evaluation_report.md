# FM-v3 evaluation report

> **Historical pre-audit ablation report.** This report evaluates the original
> `06_full_fmv3` design, not the selected final system. Several implementation
> defects were corrected afterward, and structured transition memory was then
> added. See [`fmv3_architecture_changes.md`](fmv3_architecture_changes.md) for
> the full evolution and [`structured_fmv3_report.md`](structured_fmv3_report.md)
> for the final paired result.

## Executive summary

Across paired natural-support rows, full FM-v3 changed balanced accuracy by **-0.0312**, macro-F1 by **-0.0655**, and ordinary accuracy by **-0.1365** on average relative to the re-evaluated FM-v2 checkpoint. The highest mean balanced accuracy across all evaluated rows was achieved by **FM-v2 (re-evaluated)** (0.4118). The evaluated evidence therefore does not support the full FM-v3 configuration as a replacement for FM-v2 under this protocol.

Balanced accuracy is the primary classification endpoint. Ordinary accuracy, macro-F1, coverage decomposition, calibration, and selective risk are secondary endpoints. Results are macro-averaged by event log; prefixes are never pooled across logs for the headline result.

## Implemented method

FM-v3 decouples the full support-pool label space from top-k local evidence. Its configurable head provides count-normalized local log-mean-exp evidence, global class prototypes, count-dependent prototype shrinkage, a fixed or learned global–local gate, explicit balanced/natural priors, and an uncovered-label abstention output. Pretraining mixes balanced, natural, long-tail, random-shot, missing-local-label, and missing-pool-label episodes. All target-log adaptation remains gradient-free.

The checkpoint sequence is an additive ablation: FM-v2; realistic episodes; count-neutral local evidence; global prototypes; learned shrinkage; global–local fusion; full FM-v3; no-pretraining control; and γ=0/1/learned controls.

## Evaluation protocol

- Event logs: Hospital Billing, Helpdesk, Receipt, Sepsis, and the repository's 100-case Road Traffic subset.
- Fixed case-wise held-out query set per log; support and query cases are disjoint.
- Nested absolute support budgets: 1, 2, 4, 8, 16, 32, 64, and 128 cases where available.
- Natural sampling is primary; class-aware sampling is a coverage/acquisition diagnostic.
- The query activity universe remains fixed as support shrinks; an absent support class receives zero recall.
- Repeated support sampling uses identical seeds/subsets for all methods.
- Confidence intervals resample cases, not dependent prefixes.
- The largest support pool is bounded at 128 cases for very large logs; true full-support rows are additionally included only when the support split contains at most 1,000 cases.

| log                  |   cases |   events |   activities |   median_events_per_case |
|:---------------------|--------:|---------:|-------------:|-------------------------:|
| billing              |  100000 |   451359 |           18 |                   5.0000 |
| helpdesk             |    4580 |    21348 |           14 |                   4.0000 |
| receipt              |    1434 |     8577 |           27 |                   6.0000 |
| roadtraffic100traces |     100 |      390 |           10 |                   5.0000 |
| sepsis               |    1050 |    15214 |           16 |                  13.0000 |

## Primary classification results

The following are repetition means for the configured head, balanced prior, k=20, and natural support.

| log                  | variant                         |   case_budget |   balanced_accuracy |   accuracy |   macro_f1 |
|:---------------------|:--------------------------------|--------------:|--------------------:|-----------:|-----------:|
| billing              | Count-neutral local head (γ=1)  |             1 |              0.2549 |     0.5529 |     0.2106 |
| billing              | Count-neutral local head (γ=1)  |             8 |              0.3249 |     0.6865 |     0.3048 |
| billing              | Count-neutral local head (γ=1)  |            32 |              0.3884 |     0.6442 |     0.3368 |
| billing              | Count-neutral local head (γ=1)  |           128 |              0.3776 |     0.6221 |     0.3668 |
| billing              | FM-v1-style episodic retraining |             1 |              0.2680 |     0.5846 |     0.2256 |
| billing              | FM-v1-style episodic retraining |             8 |              0.3619 |     0.8231 |     0.3501 |
| billing              | FM-v1-style episodic retraining |            32 |              0.3988 |     0.8442 |     0.3880 |
| billing              | FM-v1-style episodic retraining |           128 |              0.4552 |     0.8731 |     0.4485 |
| billing              | FM-v2 (re-evaluated)            |             1 |              0.2787 |     0.6115 |     0.2376 |
| billing              | FM-v2 (re-evaluated)            |             8 |              0.3648 |     0.8298 |     0.3514 |
| billing              | FM-v2 (re-evaluated)            |            32 |              0.4204 |     0.8500 |     0.4049 |
| billing              | FM-v2 (re-evaluated)            |           128 |              0.4851 |     0.8827 |     0.4782 |
| billing              | FM-v2 + realistic episodes      |             1 |              0.2807 |     0.6154 |     0.2392 |
| billing              | FM-v2 + realistic episodes      |             8 |              0.3648 |     0.8298 |     0.3516 |
| billing              | FM-v2 + realistic episodes      |            32 |              0.4111 |     0.8500 |     0.3959 |
| billing              | FM-v2 + realistic episodes      |           128 |              0.4824 |     0.8750 |     0.4720 |
| billing              | Full FM-v3                      |             1 |              0.2595 |     0.5644 |     0.2169 |
| billing              | Full FM-v3                      |             8 |              0.3414 |     0.7260 |     0.3245 |
| billing              | Full FM-v3                      |            32 |              0.3885 |     0.7154 |     0.3605 |
| billing              | Full FM-v3                      |           128 |              0.4564 |     0.7558 |     0.4136 |
| billing              | Full FM-v3, no pretraining      |             1 |              0.2197 |     0.4663 |     0.1769 |
| billing              | Full FM-v3, no pretraining      |             8 |              0.2818 |     0.5731 |     0.2659 |
| billing              | Full FM-v3, no pretraining      |            32 |              0.3240 |     0.6567 |     0.3256 |
| billing              | Full FM-v3, no pretraining      |           128 |              0.3946 |     0.6298 |     0.3526 |
| billing              | Global + learned shrinkage      |             1 |              0.2414 |     0.5183 |     0.1927 |
| billing              | Global + learned shrinkage      |             8 |              0.3477 |     0.7567 |     0.3328 |
| billing              | Global + learned shrinkage      |            32 |              0.3757 |     0.7606 |     0.3545 |
| billing              | Global + learned shrinkage      |           128 |              0.4652 |     0.7231 |     0.4012 |
| billing              | Global prototypes               |             1 |              0.2583 |     0.5615 |     0.2144 |
| billing              | Global prototypes               |             8 |              0.3485 |     0.7288 |     0.3229 |
| billing              | Global prototypes               |            32 |              0.3895 |     0.7087 |     0.3599 |
| billing              | Global prototypes               |           128 |              0.4509 |     0.6817 |     0.4033 |
| billing              | Global–local head               |             1 |              0.2668 |     0.5827 |     0.2239 |
| billing              | Global–local head               |             8 |              0.3422 |     0.7240 |     0.3266 |
| billing              | Global–local head               |            32 |              0.4009 |     0.7404 |     0.3813 |
| billing              | Global–local head               |           128 |              0.4863 |     0.7683 |     0.4309 |
| billing              | Local head (learned γ)          |             1 |              0.2722 |     0.5962 |     0.2315 |
| billing              | Local head (learned γ)          |             8 |              0.3289 |     0.6962 |     0.3087 |
| billing              | Local head (learned γ)          |            32 |              0.3779 |     0.6558 |     0.3349 |
| billing              | Local head (learned γ)          |           128 |              0.4126 |     0.7808 |     0.4143 |
| billing              | Local head (γ=0)                |             1 |              0.2726 |     0.5971 |     0.2318 |
| billing              | Local head (γ=0)                |             8 |              0.3175 |     0.7481 |     0.2994 |
| billing              | Local head (γ=0)                |            32 |              0.3591 |     0.8173 |     0.3435 |
| billing              | Local head (γ=0)                |           128 |              0.3949 |     0.8404 |     0.3765 |
| helpdesk             | Count-neutral local head (γ=1)  |             1 |              0.2750 |     0.5862 |     0.2323 |
| helpdesk             | Count-neutral local head (γ=1)  |             8 |              0.3068 |     0.4897 |     0.2368 |
| helpdesk             | Count-neutral local head (γ=1)  |            32 |              0.3172 |     0.4579 |     0.2565 |
| helpdesk             | Count-neutral local head (γ=1)  |           128 |              0.2854 |     0.4110 |     0.2559 |
| helpdesk             | FM-v1-style episodic retraining |             1 |              0.2859 |     0.6055 |     0.2412 |
| helpdesk             | FM-v1-style episodic retraining |             8 |              0.3091 |     0.6703 |     0.2801 |
| helpdesk             | FM-v1-style episodic retraining |            32 |              0.3378 |     0.6910 |     0.3190 |
| helpdesk             | FM-v1-style episodic retraining |           128 |              0.3456 |     0.6731 |     0.3401 |
| helpdesk             | FM-v2 (re-evaluated)            |             1 |              0.2890 |     0.6179 |     0.2480 |
| helpdesk             | FM-v2 (re-evaluated)            |             8 |              0.3009 |     0.6566 |     0.2695 |
| helpdesk             | FM-v2 (re-evaluated)            |            32 |              0.3383 |     0.6952 |     0.3206 |
| helpdesk             | FM-v2 (re-evaluated)            |           128 |              0.3225 |     0.6662 |     0.3111 |
| helpdesk             | FM-v2 + realistic episodes      |             1 |              0.2969 |     0.6400 |     0.2568 |
| helpdesk             | FM-v2 + realistic episodes      |             8 |              0.3118 |     0.6690 |     0.2817 |
| helpdesk             | FM-v2 + realistic episodes      |            32 |              0.3359 |     0.6869 |     0.3179 |
| helpdesk             | FM-v2 + realistic episodes      |           128 |              0.3308 |     0.6717 |     0.3217 |
| helpdesk             | Full FM-v3                      |             1 |              0.2785 |     0.6110 |     0.2440 |
| helpdesk             | Full FM-v3                      |             8 |              0.3078 |     0.4966 |     0.2439 |
| helpdesk             | Full FM-v3                      |            32 |              0.3294 |     0.5559 |     0.2997 |
| helpdesk             | Full FM-v3                      |           128 |              0.3345 |     0.5462 |     0.3056 |
| helpdesk             | Full FM-v3, no pretraining      |             1 |              0.1968 |     0.4317 |     0.1569 |
| helpdesk             | Full FM-v3, no pretraining      |             8 |              0.2321 |     0.3159 |     0.1698 |
| helpdesk             | Full FM-v3, no pretraining      |            32 |              0.3083 |     0.4469 |     0.2547 |
| helpdesk             | Full FM-v3, no pretraining      |           128 |              0.3108 |     0.4055 |     0.2550 |
| helpdesk             | Global + learned shrinkage      |             1 |              0.2827 |     0.5917 |     0.2356 |
| helpdesk             | Global + learned shrinkage      |             8 |              0.2674 |     0.5407 |     0.2635 |
| helpdesk             | Global + learned shrinkage      |            32 |              0.3228 |     0.5269 |     0.2919 |
| helpdesk             | Global + learned shrinkage      |           128 |              0.3133 |     0.4593 |     0.2852 |
| helpdesk             | Global prototypes               |             1 |              0.2724 |     0.5848 |     0.2330 |
| helpdesk             | Global prototypes               |             8 |              0.3304 |     0.5697 |     0.2791 |
| helpdesk             | Global prototypes               |            32 |              0.3641 |     0.5407 |     0.2973 |
| helpdesk             | Global prototypes               |           128 |              0.4152 |     0.4828 |     0.3184 |
| helpdesk             | Global–local head               |             1 |              0.2738 |     0.5945 |     0.2369 |
| helpdesk             | Global–local head               |             8 |              0.3077 |     0.4883 |     0.2448 |
| helpdesk             | Global–local head               |            32 |              0.3281 |     0.5366 |     0.2960 |
| helpdesk             | Global–local head               |           128 |              0.3238 |     0.5159 |     0.2924 |
| helpdesk             | Local head (learned γ)          |             1 |              0.2761 |     0.5890 |     0.2337 |
| helpdesk             | Local head (learned γ)          |             8 |              0.3109 |     0.4979 |     0.2434 |
| helpdesk             | Local head (learned γ)          |            32 |              0.2952 |     0.4800 |     0.2606 |
| helpdesk             | Local head (learned γ)          |           128 |              0.3115 |     0.5862 |     0.2925 |
| helpdesk             | Local head (γ=0)                |             1 |              0.2556 |     0.5517 |     0.2079 |
| helpdesk             | Local head (γ=0)                |             8 |              0.2557 |     0.6290 |     0.2158 |
| helpdesk             | Local head (γ=0)                |            32 |              0.2673 |     0.6317 |     0.2345 |
| helpdesk             | Local head (γ=0)                |           128 |              0.3107 |     0.6786 |     0.2917 |
| receipt              | Count-neutral local head (γ=1)  |             1 |              0.0816 |     0.3025 |     0.0700 |
| receipt              | Count-neutral local head (γ=1)  |             8 |              0.1921 |     0.4877 |     0.1607 |
| receipt              | Count-neutral local head (γ=1)  |            32 |              0.2846 |     0.6049 |     0.2465 |
| receipt              | Count-neutral local head (γ=1)  |           128 |              0.3864 |     0.5836 |     0.3872 |
| receipt              | FM-v1-style episodic retraining |             1 |              0.0953 |     0.3549 |     0.0798 |
| receipt              | FM-v1-style episodic retraining |             8 |              0.2392 |     0.6689 |     0.2233 |
| receipt              | FM-v1-style episodic retraining |            32 |              0.3070 |     0.7533 |     0.2976 |
| receipt              | FM-v1-style episodic retraining |           128 |              0.4445 |     0.7992 |     0.4311 |
| receipt              | FM-v2 (re-evaluated)            |             1 |              0.1083 |     0.4008 |     0.0914 |
| receipt              | FM-v2 (re-evaluated)            |             8 |              0.2375 |     0.6836 |     0.2254 |
| receipt              | FM-v2 (re-evaluated)            |            32 |              0.3113 |     0.7648 |     0.2949 |
| receipt              | FM-v2 (re-evaluated)            |           128 |              0.4505 |     0.8098 |     0.4495 |
| receipt              | FM-v2 + realistic episodes      |             1 |              0.1102 |     0.4082 |     0.0920 |
| receipt              | FM-v2 + realistic episodes      |             8 |              0.2376 |     0.6861 |     0.2260 |
| receipt              | FM-v2 + realistic episodes      |            32 |              0.3145 |     0.7730 |     0.3006 |
| receipt              | FM-v2 + realistic episodes      |           128 |              0.4382 |     0.8025 |     0.4309 |
| receipt              | Full FM-v3                      |             1 |              0.0902 |     0.3328 |     0.0763 |
| receipt              | Full FM-v3                      |             8 |              0.1905 |     0.5221 |     0.1593 |
| receipt              | Full FM-v3                      |            32 |              0.2888 |     0.5861 |     0.2305 |
| receipt              | Full FM-v3                      |           128 |              0.3928 |     0.5910 |     0.3500 |
| receipt              | Full FM-v3, no pretraining      |             1 |              0.0675 |     0.2582 |     0.0496 |
| receipt              | Full FM-v3, no pretraining      |             8 |              0.1415 |     0.3689 |     0.1152 |
| receipt              | Full FM-v3, no pretraining      |            32 |              0.1853 |     0.3934 |     0.1498 |
| receipt              | Full FM-v3, no pretraining      |           128 |              0.2108 |     0.4115 |     0.1998 |
| receipt              | Global + learned shrinkage      |             1 |              0.0851 |     0.3246 |     0.0699 |
| receipt              | Global + learned shrinkage      |             8 |              0.1586 |     0.4238 |     0.1293 |
| receipt              | Global + learned shrinkage      |            32 |              0.2156 |     0.4393 |     0.1650 |
| receipt              | Global + learned shrinkage      |           128 |              0.2946 |     0.4205 |     0.2370 |
| receipt              | Global prototypes               |             1 |              0.0822 |     0.3057 |     0.0699 |
| receipt              | Global prototypes               |             8 |              0.1767 |     0.4746 |     0.1518 |
| receipt              | Global prototypes               |            32 |              0.2504 |     0.4844 |     0.2003 |
| receipt              | Global prototypes               |           128 |              0.2818 |     0.4541 |     0.2529 |
| receipt              | Global–local head               |             1 |              0.0893 |     0.3303 |     0.0754 |
| receipt              | Global–local head               |             8 |              0.1893 |     0.5008 |     0.1581 |
| receipt              | Global–local head               |            32 |              0.2857 |     0.5582 |     0.2232 |
| receipt              | Global–local head               |           128 |              0.3933 |     0.5623 |     0.3429 |
| receipt              | Local head (learned γ)          |             1 |              0.0837 |     0.3090 |     0.0722 |
| receipt              | Local head (learned γ)          |             8 |              0.1896 |     0.5049 |     0.1651 |
| receipt              | Local head (learned γ)          |            32 |              0.2933 |     0.6672 |     0.2626 |
| receipt              | Local head (learned γ)          |           128 |              0.4128 |     0.7213 |     0.4182 |
| receipt              | Local head (γ=0)                |             1 |              0.0849 |     0.3156 |     0.0727 |
| receipt              | Local head (γ=0)                |             8 |              0.1026 |     0.3943 |     0.0861 |
| receipt              | Local head (γ=0)                |            32 |              0.1553 |     0.5877 |     0.1417 |
| receipt              | Local head (γ=0)                |           128 |              0.2293 |     0.7148 |     0.2162 |
| roadtraffic100traces | Count-neutral local head (γ=1)  |             1 |              0.5914 |     0.6271 |     0.5108 |
| roadtraffic100traces | Count-neutral local head (γ=1)  |             8 |              0.6188 |     0.6441 |     0.5912 |
| roadtraffic100traces | Count-neutral local head (γ=1)  |            32 |              0.4314 |     0.4203 |     0.4500 |
| roadtraffic100traces | Count-neutral local head (γ=1)  |            43 |              0.5360 |     0.5254 |     0.5489 |
| roadtraffic100traces | FM-v1-style episodic retraining |             1 |              0.5874 |     0.6102 |     0.4969 |
| roadtraffic100traces | FM-v1-style episodic retraining |             8 |              0.7138 |     0.7458 |     0.6619 |
| roadtraffic100traces | FM-v1-style episodic retraining |            32 |              0.7833 |     0.8169 |     0.7743 |
| roadtraffic100traces | FM-v1-style episodic retraining |            43 |              0.7532 |     0.7797 |     0.7452 |
| roadtraffic100traces | FM-v2 (re-evaluated)            |             1 |              0.5929 |     0.6169 |     0.5025 |
| roadtraffic100traces | FM-v2 (re-evaluated)            |             8 |              0.7610 |     0.7898 |     0.7324 |
| roadtraffic100traces | FM-v2 (re-evaluated)            |            32 |              0.7845 |     0.8169 |     0.7715 |
| roadtraffic100traces | FM-v2 (re-evaluated)            |            43 |              0.7828 |     0.8136 |     0.7629 |
| roadtraffic100traces | FM-v2 + realistic episodes      |             1 |              0.5941 |     0.6136 |     0.5071 |
| roadtraffic100traces | FM-v2 + realistic episodes      |             8 |              0.7426 |     0.7695 |     0.7134 |
| roadtraffic100traces | FM-v2 + realistic episodes      |            32 |              0.7771 |     0.8102 |     0.7650 |
| roadtraffic100traces | FM-v2 + realistic episodes      |            43 |              0.8056 |     0.8305 |     0.7808 |
| roadtraffic100traces | Full FM-v3                      |             1 |              0.6096 |     0.6407 |     0.5251 |
| roadtraffic100traces | Full FM-v3                      |             8 |              0.5854 |     0.6034 |     0.5423 |
| roadtraffic100traces | Full FM-v3                      |            32 |              0.7321 |     0.7525 |     0.7272 |
| roadtraffic100traces | Full FM-v3                      |            43 |              0.7273 |     0.7458 |     0.7201 |
| roadtraffic100traces | Full FM-v3, no pretraining      |             1 |              0.5682 |     0.5932 |     0.4817 |
| roadtraffic100traces | Full FM-v3, no pretraining      |             8 |              0.5654 |     0.5898 |     0.4998 |
| roadtraffic100traces | Full FM-v3, no pretraining      |            32 |              0.6351 |     0.6576 |     0.6265 |
| roadtraffic100traces | Full FM-v3, no pretraining      |            43 |              0.6212 |     0.6610 |     0.5718 |
| roadtraffic100traces | Global + learned shrinkage      |             1 |              0.5763 |     0.5966 |     0.4864 |
| roadtraffic100traces | Global + learned shrinkage      |             8 |              0.6394 |     0.6780 |     0.5926 |
| roadtraffic100traces | Global + learned shrinkage      |            32 |              0.6379 |     0.6746 |     0.5938 |
| roadtraffic100traces | Global + learned shrinkage      |            43 |              0.6212 |     0.6610 |     0.5662 |
| roadtraffic100traces | Global prototypes               |             1 |              0.5706 |     0.6000 |     0.4928 |
| roadtraffic100traces | Global prototypes               |             8 |              0.5904 |     0.6136 |     0.5531 |
| roadtraffic100traces | Global prototypes               |            32 |              0.4851 |     0.4915 |     0.4599 |
| roadtraffic100traces | Global prototypes               |            43 |              0.4823 |     0.4915 |     0.4613 |
| roadtraffic100traces | Global–local head               |             1 |              0.5874 |     0.6102 |     0.4969 |
| roadtraffic100traces | Global–local head               |             8 |              0.6348 |     0.6576 |     0.5882 |
| roadtraffic100traces | Global–local head               |            32 |              0.7182 |     0.7390 |     0.7115 |
| roadtraffic100traces | Global–local head               |            43 |              0.7273 |     0.7458 |     0.7201 |
| roadtraffic100traces | Local head (learned γ)          |             1 |              0.5914 |     0.6271 |     0.5108 |
| roadtraffic100traces | Local head (learned γ)          |             8 |              0.5785 |     0.5966 |     0.5261 |
| roadtraffic100traces | Local head (learned γ)          |            32 |              0.7598 |     0.7898 |     0.7450 |
| roadtraffic100traces | Local head (learned γ)          |            43 |              0.7689 |     0.7966 |     0.7515 |
| roadtraffic100traces | Local head (γ=0)                |             1 |              0.5914 |     0.6271 |     0.5108 |
| roadtraffic100traces | Local head (γ=0)                |             8 |              0.5880 |     0.6508 |     0.5160 |
| roadtraffic100traces | Local head (γ=0)                |            32 |              0.6920 |     0.7356 |     0.6385 |
| roadtraffic100traces | Local head (γ=0)                |            43 |              0.6629 |     0.7119 |     0.5843 |
| sepsis               | Count-neutral local head (γ=1)  |             1 |              0.2275 |     0.2787 |     0.1562 |
| sepsis               | Count-neutral local head (γ=1)  |             8 |              0.3147 |     0.3750 |     0.2946 |
| sepsis               | Count-neutral local head (γ=1)  |            32 |              0.3198 |     0.3284 |     0.2903 |
| sepsis               | Count-neutral local head (γ=1)  |           128 |              0.4195 |     0.3305 |     0.3697 |
| sepsis               | Count-neutral local head (γ=1)  |          1000 |              0.4691 |     0.3404 |     0.4105 |
| sepsis               | FM-v1-style episodic retraining |             1 |              0.2522 |     0.3922 |     0.2207 |
| sepsis               | FM-v1-style episodic retraining |             8 |              0.3337 |     0.5069 |     0.3423 |
| sepsis               | FM-v1-style episodic retraining |            32 |              0.4369 |     0.5612 |     0.4445 |
| sepsis               | FM-v1-style episodic retraining |           128 |              0.5476 |     0.5838 |     0.5538 |
| sepsis               | FM-v1-style episodic retraining |          1000 |              0.6007 |     0.5926 |     0.5990 |
| sepsis               | FM-v2 (re-evaluated)            |             1 |              0.2621 |     0.4088 |     0.2317 |
| sepsis               | FM-v2 (re-evaluated)            |             8 |              0.3512 |     0.5390 |     0.3607 |
| sepsis               | FM-v2 (re-evaluated)            |            32 |              0.4421 |     0.5630 |     0.4487 |
| sepsis               | FM-v2 (re-evaluated)            |           128 |              0.5415 |     0.5901 |     0.5489 |
| sepsis               | FM-v2 (re-evaluated)            |          1000 |              0.6005 |     0.5926 |     0.5960 |
| sepsis               | FM-v2 + realistic episodes      |             1 |              0.2609 |     0.4014 |     0.2274 |
| sepsis               | FM-v2 + realistic episodes      |             8 |              0.3499 |     0.5354 |     0.3562 |
| sepsis               | FM-v2 + realistic episodes      |            32 |              0.4296 |     0.5612 |     0.4289 |
| sepsis               | FM-v2 + realistic episodes      |           128 |              0.5374 |     0.5908 |     0.5458 |
| sepsis               | FM-v2 + realistic episodes      |          1000 |              0.6041 |     0.6049 |     0.5994 |
| sepsis               | Full FM-v3                      |             1 |              0.2304 |     0.2811 |     0.1572 |
| sepsis               | Full FM-v3                      |             8 |              0.3075 |     0.2780 |     0.2223 |
| sepsis               | Full FM-v3                      |            32 |              0.4258 |     0.2966 |     0.2784 |
| sepsis               | Full FM-v3                      |           128 |              0.5210 |     0.3347 |     0.3512 |
| sepsis               | Full FM-v3                      |          1000 |              0.5615 |     0.3580 |     0.4357 |
| sepsis               | Full FM-v3, no pretraining      |             1 |              0.2037 |     0.2466 |     0.1356 |
| sepsis               | Full FM-v3, no pretraining      |             8 |              0.2540 |     0.2205 |     0.1703 |
| sepsis               | Full FM-v3, no pretraining      |            32 |              0.3414 |     0.2462 |     0.2161 |
| sepsis               | Full FM-v3, no pretraining      |           128 |              0.4570 |     0.2734 |     0.2903 |
| sepsis               | Full FM-v3, no pretraining      |          1000 |              0.4111 |     0.2892 |     0.3464 |
| sepsis               | Global + learned shrinkage      |             1 |              0.2186 |     0.2765 |     0.1542 |
| sepsis               | Global + learned shrinkage      |             8 |              0.2711 |     0.2462 |     0.1876 |
| sepsis               | Global + learned shrinkage      |            32 |              0.3860 |     0.2533 |     0.2277 |
| sepsis               | Global + learned shrinkage      |           128 |              0.4701 |     0.2494 |     0.2719 |
| sepsis               | Global + learned shrinkage      |          1000 |              0.4621 |     0.2698 |     0.3184 |
| sepsis               | Global prototypes               |             1 |              0.2347 |     0.3065 |     0.1751 |
| sepsis               | Global prototypes               |             8 |              0.2971 |     0.2515 |     0.2058 |
| sepsis               | Global prototypes               |            32 |              0.3443 |     0.2547 |     0.2473 |
| sepsis               | Global prototypes               |           128 |              0.4223 |     0.2522 |     0.2997 |
| sepsis               | Global prototypes               |          1000 |              0.4407 |     0.2646 |     0.3012 |
| sepsis               | Global–local head               |             1 |              0.2359 |     0.2854 |     0.1623 |
| sepsis               | Global–local head               |             8 |              0.3015 |     0.2695 |     0.2191 |
| sepsis               | Global–local head               |            32 |              0.4263 |     0.3034 |     0.2723 |
| sepsis               | Global–local head               |           128 |              0.5180 |     0.3224 |     0.3361 |
| sepsis               | Global–local head               |          1000 |              0.5630 |     0.3668 |     0.4340 |
| sepsis               | Local head (learned γ)          |             1 |              0.2305 |     0.2818 |     0.1584 |
| sepsis               | Local head (learned γ)          |             8 |              0.3151 |     0.3792 |     0.2949 |
| sepsis               | Local head (learned γ)          |            32 |              0.3883 |     0.4222 |     0.3578 |
| sepsis               | Local head (learned γ)          |           128 |              0.5164 |     0.4797 |     0.4757 |
| sepsis               | Local head (learned γ)          |          1000 |              0.5599 |     0.5520 |     0.5624 |
| sepsis               | Local head (γ=0)                |             1 |              0.1732 |     0.3414 |     0.1348 |
| sepsis               | Local head (γ=0)                |             8 |              0.2361 |     0.3919 |     0.1796 |
| sepsis               | Local head (γ=0)                |            32 |              0.3112 |     0.5037 |     0.2995 |
| sepsis               | Local head (γ=0)                |           128 |              0.3572 |     0.5605 |     0.3627 |
| sepsis               | Local head (γ=0)                |          1000 |              0.5484 |     0.6085 |     0.5617 |

### Macro-average learning curves

| variant                         |   case_budget |   n_logs |   balanced_accuracy |   accuracy |   macro_f1 |
|:--------------------------------|--------------:|---------:|--------------------:|-----------:|-----------:|
| Count-neutral local head (γ=1)  |             1 |        5 |              0.2861 |     0.4695 |     0.2360 |
| Count-neutral local head (γ=1)  |             2 |        5 |              0.3128 |     0.5276 |     0.2683 |
| Count-neutral local head (γ=1)  |             4 |        5 |              0.3365 |     0.5206 |     0.2923 |
| Count-neutral local head (γ=1)  |             8 |        5 |              0.3515 |     0.5366 |     0.3176 |
| Count-neutral local head (γ=1)  |            16 |        5 |              0.3719 |     0.5390 |     0.3323 |
| Count-neutral local head (γ=1)  |            32 |        5 |              0.3483 |     0.4912 |     0.3160 |
| Count-neutral local head (γ=1)  |            43 |        1 |              0.5360 |     0.5254 |     0.5489 |
| Count-neutral local head (γ=1)  |            64 |        4 |              0.3730 |     0.4816 |     0.3232 |
| Count-neutral local head (γ=1)  |           128 |        4 |              0.3672 |     0.4868 |     0.3449 |
| Count-neutral local head (γ=1)  |          1000 |        1 |              0.4691 |     0.3404 |     0.4105 |
| FM-v1-style episodic retraining |             1 |        5 |              0.2977 |     0.5095 |     0.2528 |
| FM-v1-style episodic retraining |             2 |        5 |              0.3284 |     0.5867 |     0.2964 |
| FM-v1-style episodic retraining |             4 |        5 |              0.3639 |     0.6350 |     0.3337 |
| FM-v1-style episodic retraining |             8 |        5 |              0.3915 |     0.6830 |     0.3715 |
| FM-v1-style episodic retraining |            16 |        5 |              0.4323 |     0.7264 |     0.4193 |
| FM-v1-style episodic retraining |            32 |        5 |              0.4528 |     0.7333 |     0.4447 |
| FM-v1-style episodic retraining |            43 |        1 |              0.7532 |     0.7797 |     0.7452 |
| FM-v1-style episodic retraining |            64 |        4 |              0.4113 |     0.7267 |     0.4058 |
| FM-v1-style episodic retraining |           128 |        4 |              0.4482 |     0.7323 |     0.4434 |
| FM-v1-style episodic retraining |          1000 |        1 |              0.6007 |     0.5926 |     0.5990 |
| FM-v2 (re-evaluated)            |             1 |        5 |              0.3062 |     0.5312 |     0.2622 |
| FM-v2 (re-evaluated)            |             2 |        5 |              0.3372 |     0.6067 |     0.3044 |
| FM-v2 (re-evaluated)            |             4 |        5 |              0.3682 |     0.6459 |     0.3396 |
| FM-v2 (re-evaluated)            |             8 |        5 |              0.4031 |     0.6998 |     0.3879 |
| FM-v2 (re-evaluated)            |            16 |        5 |              0.4439 |     0.7358 |     0.4281 |
| FM-v2 (re-evaluated)            |            32 |        5 |              0.4593 |     0.7380 |     0.4481 |
| FM-v2 (re-evaluated)            |            43 |        1 |              0.7828 |     0.8136 |     0.7629 |
| FM-v2 (re-evaluated)            |            64 |        4 |              0.4250 |     0.7273 |     0.4179 |
| FM-v2 (re-evaluated)            |           128 |        4 |              0.4499 |     0.7372 |     0.4469 |
| FM-v2 (re-evaluated)            |          1000 |        1 |              0.6005 |     0.5926 |     0.5960 |
| FM-v2 + realistic episodes      |             1 |        5 |              0.3085 |     0.5357 |     0.2645 |
| FM-v2 + realistic episodes      |             2 |        5 |              0.3336 |     0.6017 |     0.3013 |
| FM-v2 + realistic episodes      |             4 |        5 |              0.3684 |     0.6439 |     0.3391 |
| FM-v2 + realistic episodes      |             8 |        5 |              0.4013 |     0.6980 |     0.3858 |
| FM-v2 + realistic episodes      |            16 |        5 |              0.4400 |     0.7342 |     0.4264 |
| FM-v2 + realistic episodes      |            32 |        5 |              0.4537 |     0.7362 |     0.4417 |
| FM-v2 + realistic episodes      |            43 |        1 |              0.8056 |     0.8305 |     0.7808 |
| FM-v2 + realistic episodes      |            64 |        4 |              0.4298 |     0.7289 |     0.4204 |
| FM-v2 + realistic episodes      |           128 |        4 |              0.4472 |     0.7350 |     0.4426 |
| FM-v2 + realistic episodes      |          1000 |        1 |              0.6041 |     0.6049 |     0.5994 |
| Full FM-v3                      |             1 |        5 |              0.2936 |     0.4860 |     0.2439 |
| Full FM-v3                      |             2 |        5 |              0.3106 |     0.5280 |     0.2646 |
| Full FM-v3                      |             4 |        5 |              0.3313 |     0.5170 |     0.2799 |
| Full FM-v3                      |             8 |        5 |              0.3465 |     0.5252 |     0.2985 |
| Full FM-v3                      |            16 |        5 |              0.4152 |     0.5741 |     0.3607 |
| Full FM-v3                      |            32 |        5 |              0.4329 |     0.5813 |     0.3793 |
| Full FM-v3                      |            43 |        1 |              0.7273 |     0.7458 |     0.7201 |
| Full FM-v3                      |            64 |        4 |              0.3950 |     0.5506 |     0.3346 |
| Full FM-v3                      |           128 |        4 |              0.4262 |     0.5569 |     0.3551 |
| Full FM-v3                      |          1000 |        1 |              0.5615 |     0.3580 |     0.4357 |
| Full FM-v3, no pretraining      |             1 |        5 |              0.2512 |     0.3992 |     0.2001 |
| Full FM-v3, no pretraining      |             2 |        5 |              0.2603 |     0.4243 |     0.2142 |
| Full FM-v3, no pretraining      |             4 |        5 |              0.2755 |     0.4053 |     0.2210 |
| Full FM-v3, no pretraining      |             8 |        5 |              0.2950 |     0.4136 |     0.2442 |
| Full FM-v3, no pretraining      |            16 |        5 |              0.3193 |     0.4404 |     0.2796 |
| Full FM-v3, no pretraining      |            32 |        5 |              0.3588 |     0.4802 |     0.3146 |
| Full FM-v3, no pretraining      |            43 |        1 |              0.6212 |     0.6610 |     0.5718 |
| Full FM-v3, no pretraining      |            64 |        4 |              0.3129 |     0.4180 |     0.2529 |
| Full FM-v3, no pretraining      |           128 |        4 |              0.3433 |     0.4300 |     0.2744 |
| Full FM-v3, no pretraining      |          1000 |        1 |              0.4111 |     0.2892 |     0.3464 |
| Global + learned shrinkage      |             1 |        5 |              0.2808 |     0.4615 |     0.2277 |
| Global + learned shrinkage      |             2 |        5 |              0.2972 |     0.5057 |     0.2536 |
| Global + learned shrinkage      |             4 |        5 |              0.3239 |     0.5072 |     0.2734 |
| Global + learned shrinkage      |             8 |        5 |              0.3369 |     0.5291 |     0.3012 |
| Global + learned shrinkage      |            16 |        5 |              0.3584 |     0.5487 |     0.3193 |
| Global + learned shrinkage      |            32 |        5 |              0.3876 |     0.5309 |     0.3266 |
| Global + learned shrinkage      |            43 |        1 |              0.6212 |     0.6610 |     0.5662 |
| Global + learned shrinkage      |            64 |        4 |              0.3269 |     0.4670 |     0.2700 |
| Global + learned shrinkage      |           128 |        4 |              0.3858 |     0.4631 |     0.2988 |
| Global + learned shrinkage      |          1000 |        1 |              0.4621 |     0.2698 |     0.3184 |
| Global prototypes               |             1 |        5 |              0.2836 |     0.4717 |     0.2371 |
| Global prototypes               |             2 |        5 |              0.3031 |     0.5196 |     0.2602 |
| Global prototypes               |             4 |        5 |              0.3372 |     0.5232 |     0.2860 |
| Global prototypes               |             8 |        5 |              0.3486 |     0.5276 |     0.3026 |
| Global prototypes               |            16 |        5 |              0.3640 |     0.5188 |     0.3071 |
| Global prototypes               |            32 |        5 |              0.3667 |     0.4960 |     0.3129 |
| Global prototypes               |            43 |        1 |              0.4823 |     0.4915 |     0.4613 |
| Global prototypes               |            64 |        4 |              0.3624 |     0.4729 |     0.2991 |
| Global prototypes               |           128 |        4 |              0.3925 |     0.4677 |     0.3186 |
| Global prototypes               |          1000 |        1 |              0.4407 |     0.2646 |     0.3012 |
| Global–local head               |             1 |        5 |              0.2906 |     0.4806 |     0.2391 |
| Global–local head               |             2 |        5 |              0.3180 |     0.5382 |     0.2738 |
| Global–local head               |             4 |        5 |              0.3315 |     0.5105 |     0.2777 |
| Global–local head               |             8 |        5 |              0.3551 |     0.5280 |     0.3074 |
| Global–local head               |            16 |        5 |              0.4075 |     0.5705 |     0.3544 |
| Global–local head               |            32 |        5 |              0.4318 |     0.5755 |     0.3769 |
| Global–local head               |            43 |        1 |              0.7273 |     0.7458 |     0.7201 |
| Global–local head               |            64 |        4 |              0.3895 |     0.5400 |     0.3244 |
| Global–local head               |           128 |        4 |              0.4303 |     0.5422 |     0.3506 |
| Global–local head               |          1000 |        1 |              0.5630 |     0.3668 |     0.4340 |
| Local head (learned γ)          |             1 |        5 |              0.2908 |     0.4806 |     0.2413 |
| Local head (learned γ)          |             2 |        5 |              0.3140 |     0.5330 |     0.2701 |
| Local head (learned γ)          |             4 |        5 |              0.3386 |     0.5264 |     0.2926 |
| Local head (learned γ)          |             8 |        5 |              0.3446 |     0.5350 |     0.3077 |
| Local head (learned γ)          |            16 |        5 |              0.4123 |     0.5984 |     0.3713 |
| Local head (learned γ)          |            32 |        5 |              0.4229 |     0.6030 |     0.3922 |
| Local head (learned γ)          |            43 |        1 |              0.7689 |     0.7966 |     0.7515 |
| Local head (learned γ)          |            64 |        4 |              0.3893 |     0.6021 |     0.3614 |
| Local head (learned γ)          |           128 |        4 |              0.4133 |     0.6420 |     0.4002 |
| Local head (learned γ)          |          1000 |        1 |              0.5599 |     0.5520 |     0.5624 |
| Local head (γ=0)                |             1 |        5 |              0.2755 |     0.4866 |     0.2316 |
| Local head (γ=0)                |             2 |        5 |              0.2740 |     0.5162 |     0.2317 |
| Local head (γ=0)                |             4 |        5 |              0.2666 |     0.5112 |     0.2249 |
| Local head (γ=0)                |             8 |        5 |              0.3000 |     0.5628 |     0.2594 |
| Local head (γ=0)                |            16 |        5 |              0.3335 |     0.6147 |     0.2989 |
| Local head (γ=0)                |            32 |        5 |              0.3570 |     0.6552 |     0.3316 |
| Local head (γ=0)                |            43 |        1 |              0.6629 |     0.7119 |     0.5843 |
| Local head (γ=0)                |            64 |        4 |              0.2998 |     0.6754 |     0.2872 |
| Local head (γ=0)                |           128 |        4 |              0.3230 |     0.6986 |     0.3118 |
| Local head (γ=0)                |          1000 |        1 |              0.5484 |     0.6085 |     0.5617 |

### Lower-quartile and worst-log performance

Each log contributes equally. `lower_quartile_log` and `worst_log` are computed after averaging repetitions within each event log.

| variant                         |   case_budget |   n_logs |   mean_log |   lower_quartile_log |   worst_log |
|:--------------------------------|--------------:|---------:|-----------:|---------------------:|------------:|
| Count-neutral local head (γ=1)  |             1 |        5 |     0.2861 |               0.2275 |      0.0816 |
| Count-neutral local head (γ=1)  |             2 |        5 |     0.3128 |               0.2603 |      0.1034 |
| Count-neutral local head (γ=1)  |             4 |        5 |     0.3365 |               0.2730 |      0.1208 |
| Count-neutral local head (γ=1)  |             8 |        5 |     0.3515 |               0.3068 |      0.1921 |
| Count-neutral local head (γ=1)  |            16 |        5 |     0.3719 |               0.3256 |      0.2592 |
| Count-neutral local head (γ=1)  |            32 |        5 |     0.3483 |               0.3172 |      0.2846 |
| Count-neutral local head (γ=1)  |            43 |        1 |     0.5360 |               0.5360 |      0.5360 |
| Count-neutral local head (γ=1)  |            64 |        4 |     0.3730 |               0.3421 |      0.3356 |
| Count-neutral local head (γ=1)  |           128 |        4 |     0.3672 |               0.3545 |      0.2854 |
| Count-neutral local head (γ=1)  |          1000 |        1 |     0.4691 |               0.4691 |      0.4691 |
| FM-v1-style episodic retraining |             1 |        5 |     0.2977 |               0.2522 |      0.0953 |
| FM-v1-style episodic retraining |             2 |        5 |     0.3284 |               0.2621 |      0.1281 |
| FM-v1-style episodic retraining |             4 |        5 |     0.3639 |               0.3001 |      0.1669 |
| FM-v1-style episodic retraining |             8 |        5 |     0.3915 |               0.3091 |      0.2392 |
| FM-v1-style episodic retraining |            16 |        5 |     0.4323 |               0.3166 |      0.2923 |
| FM-v1-style episodic retraining |            32 |        5 |     0.4528 |               0.3378 |      0.3070 |
| FM-v1-style episodic retraining |            43 |        1 |     0.7532 |               0.7532 |      0.7532 |
| FM-v1-style episodic retraining |            64 |        4 |     0.4113 |               0.3722 |      0.3459 |
| FM-v1-style episodic retraining |           128 |        4 |     0.4482 |               0.4198 |      0.3456 |
| FM-v1-style episodic retraining |          1000 |        1 |     0.6007 |               0.6007 |      0.6007 |
| FM-v2 (re-evaluated)            |             1 |        5 |     0.3062 |               0.2621 |      0.1083 |
| FM-v2 (re-evaluated)            |             2 |        5 |     0.3372 |               0.2717 |      0.1409 |
| FM-v2 (re-evaluated)            |             4 |        5 |     0.3682 |               0.2916 |      0.1701 |
| FM-v2 (re-evaluated)            |             8 |        5 |     0.4031 |               0.3009 |      0.2375 |
| FM-v2 (re-evaluated)            |            16 |        5 |     0.4439 |               0.3167 |      0.3049 |
| FM-v2 (re-evaluated)            |            32 |        5 |     0.4593 |               0.3383 |      0.3113 |
| FM-v2 (re-evaluated)            |            43 |        1 |     0.7828 |               0.7828 |      0.7828 |
| FM-v2 (re-evaluated)            |            64 |        4 |     0.4250 |               0.3766 |      0.3345 |
| FM-v2 (re-evaluated)            |           128 |        4 |     0.4499 |               0.4185 |      0.3225 |
| FM-v2 (re-evaluated)            |          1000 |        1 |     0.6005 |               0.6005 |      0.6005 |
| FM-v2 + realistic episodes      |             1 |        5 |     0.3085 |               0.2609 |      0.1102 |
| FM-v2 + realistic episodes      |             2 |        5 |     0.3336 |               0.2566 |      0.1387 |
| FM-v2 + realistic episodes      |             4 |        5 |     0.3684 |               0.2959 |      0.1728 |
| FM-v2 + realistic episodes      |             8 |        5 |     0.4013 |               0.3118 |      0.2376 |
| FM-v2 + realistic episodes      |            16 |        5 |     0.4400 |               0.3230 |      0.2961 |
| FM-v2 + realistic episodes      |            32 |        5 |     0.4537 |               0.3359 |      0.3145 |
| FM-v2 + realistic episodes      |            43 |        1 |     0.8056 |               0.8056 |      0.8056 |
| FM-v2 + realistic episodes      |            64 |        4 |     0.4298 |               0.3748 |      0.3357 |
| FM-v2 + realistic episodes      |           128 |        4 |     0.4472 |               0.4114 |      0.3308 |
| FM-v2 + realistic episodes      |          1000 |        1 |     0.6041 |               0.6041 |      0.6041 |
| Full FM-v3                      |             1 |        5 |     0.2936 |               0.2304 |      0.0902 |
| Full FM-v3                      |             2 |        5 |     0.3106 |               0.2495 |      0.1043 |
| Full FM-v3                      |             4 |        5 |     0.3313 |               0.2540 |      0.1244 |
| Full FM-v3                      |             8 |        5 |     0.3465 |               0.3075 |      0.1905 |
| Full FM-v3                      |            16 |        5 |     0.4152 |               0.3302 |      0.2673 |
| Full FM-v3                      |            32 |        5 |     0.4329 |               0.3294 |      0.2888 |
| Full FM-v3                      |            43 |        1 |     0.7273 |               0.7273 |      0.7273 |
| Full FM-v3                      |            64 |        4 |     0.3950 |               0.3585 |      0.3257 |
| Full FM-v3                      |           128 |        4 |     0.4262 |               0.3783 |      0.3345 |
| Full FM-v3                      |          1000 |        1 |     0.5615 |               0.5615 |      0.5615 |
| Full FM-v3, no pretraining      |             1 |        5 |     0.2512 |               0.1968 |      0.0675 |
| Full FM-v3, no pretraining      |             2 |        5 |     0.2603 |               0.1966 |      0.0797 |
| Full FM-v3, no pretraining      |             4 |        5 |     0.2755 |               0.1971 |      0.0783 |
| Full FM-v3, no pretraining      |             8 |        5 |     0.2950 |               0.2321 |      0.1415 |
| Full FM-v3, no pretraining      |            16 |        5 |     0.3193 |               0.2788 |      0.1676 |
| Full FM-v3, no pretraining      |            32 |        5 |     0.3588 |               0.3083 |      0.1853 |
| Full FM-v3, no pretraining      |            43 |        1 |     0.6212 |               0.6212 |      0.6212 |
| Full FM-v3, no pretraining      |            64 |        4 |     0.3129 |               0.2791 |      0.2226 |
| Full FM-v3, no pretraining      |           128 |        4 |     0.3433 |               0.2858 |      0.2108 |
| Full FM-v3, no pretraining      |          1000 |        1 |     0.4111 |               0.4111 |      0.4111 |
| Global + learned shrinkage      |             1 |        5 |     0.2808 |               0.2186 |      0.0851 |
| Global + learned shrinkage      |             2 |        5 |     0.2972 |               0.2211 |      0.0986 |
| Global + learned shrinkage      |             4 |        5 |     0.3239 |               0.2371 |      0.1156 |
| Global + learned shrinkage      |             8 |        5 |     0.3369 |               0.2674 |      0.1586 |
| Global + learned shrinkage      |            16 |        5 |     0.3584 |               0.2789 |      0.2086 |
| Global + learned shrinkage      |            32 |        5 |     0.3876 |               0.3228 |      0.2156 |
| Global + learned shrinkage      |            43 |        1 |     0.6212 |               0.6212 |      0.6212 |
| Global + learned shrinkage      |            64 |        4 |     0.3269 |               0.2780 |      0.2528 |
| Global + learned shrinkage      |           128 |        4 |     0.3858 |               0.3086 |      0.2946 |
| Global + learned shrinkage      |          1000 |        1 |     0.4621 |               0.4621 |      0.4621 |
| Global prototypes               |             1 |        5 |     0.2836 |               0.2347 |      0.0822 |
| Global prototypes               |             2 |        5 |     0.3031 |               0.2505 |      0.1014 |
| Global prototypes               |             4 |        5 |     0.3372 |               0.2856 |      0.1230 |
| Global prototypes               |             8 |        5 |     0.3486 |               0.2971 |      0.1767 |
| Global prototypes               |            16 |        5 |     0.3640 |               0.3246 |      0.2281 |
| Global prototypes               |            32 |        5 |     0.3667 |               0.3443 |      0.2504 |
| Global prototypes               |            43 |        1 |     0.4823 |               0.4823 |      0.4823 |
| Global prototypes               |            64 |        4 |     0.3624 |               0.3323 |      0.2750 |
| Global prototypes               |           128 |        4 |     0.3925 |               0.3818 |      0.2818 |
| Global prototypes               |          1000 |        1 |     0.4407 |               0.4407 |      0.4407 |
| Global–local head               |             1 |        5 |     0.2906 |               0.2359 |      0.0893 |
| Global–local head               |             2 |        5 |     0.3180 |               0.2519 |      0.1045 |
| Global–local head               |             4 |        5 |     0.3315 |               0.2628 |      0.1177 |
| Global–local head               |             8 |        5 |     0.3551 |               0.3015 |      0.1893 |
| Global–local head               |            16 |        5 |     0.4075 |               0.3297 |      0.2653 |
| Global–local head               |            32 |        5 |     0.4318 |               0.3281 |      0.2857 |
| Global–local head               |            43 |        1 |     0.7273 |               0.7273 |      0.7273 |
| Global–local head               |            64 |        4 |     0.3895 |               0.3529 |      0.3152 |
| Global–local head               |           128 |        4 |     0.4303 |               0.3759 |      0.3238 |
| Global–local head               |          1000 |        1 |     0.5630 |               0.5630 |      0.5630 |
| Local head (learned γ)          |             1 |        5 |     0.2908 |               0.2305 |      0.0837 |
| Local head (learned γ)          |             2 |        5 |     0.3140 |               0.2608 |      0.1044 |
| Local head (learned γ)          |             4 |        5 |     0.3386 |               0.2727 |      0.1243 |
| Local head (learned γ)          |             8 |        5 |     0.3446 |               0.3109 |      0.1896 |
| Local head (learned γ)          |            16 |        5 |     0.4123 |               0.3455 |      0.2631 |
| Local head (learned γ)          |            32 |        5 |     0.4229 |               0.2952 |      0.2933 |
| Local head (learned γ)          |            43 |        1 |     0.7689 |               0.7689 |      0.7689 |
| Local head (learned γ)          |            64 |        4 |     0.3893 |               0.3467 |      0.2804 |
| Local head (learned γ)          |           128 |        4 |     0.4133 |               0.3873 |      0.3115 |
| Local head (learned γ)          |          1000 |        1 |     0.5599 |               0.5599 |      0.5599 |
| Local head (γ=0)                |             1 |        5 |     0.2755 |               0.1732 |      0.0849 |
| Local head (γ=0)                |             2 |        5 |     0.2740 |               0.1183 |      0.0987 |
| Local head (γ=0)                |             4 |        5 |     0.2666 |               0.1600 |      0.0919 |
| Local head (γ=0)                |             8 |        5 |     0.3000 |               0.2361 |      0.1026 |
| Local head (γ=0)                |            16 |        5 |     0.3335 |               0.2541 |      0.1304 |
| Local head (γ=0)                |            32 |        5 |     0.3570 |               0.2673 |      0.1553 |
| Local head (γ=0)                |            43 |        1 |     0.6629 |               0.6629 |      0.6629 |
| Local head (γ=0)                |            64 |        4 |     0.2998 |               0.2687 |      0.1978 |
| Local head (γ=0)                |           128 |        4 |     0.3230 |               0.2903 |      0.2293 |
| Local head (γ=0)                |          1000 |        1 |     0.5484 |               0.5484 |      0.5484 |

### Case-level uncertainty

Intervals are percentile intervals formed by resampling complete query cases, preserving dependence among prefixes from the same case. The table averages row-level interval endpoints across logs and repetitions.

| variant              |   case_budget |   balanced_accuracy |   ci_lower |   ci_upper |
|:---------------------|--------------:|--------------------:|-----------:|-----------:|
| FM-v2 (re-evaluated) |             1 |              0.3062 |     0.2704 |     0.3353 |
| FM-v2 (re-evaluated) |             2 |              0.3372 |     0.2980 |     0.3729 |
| FM-v2 (re-evaluated) |             4 |              0.3682 |     0.3264 |     0.4068 |
| FM-v2 (re-evaluated) |             8 |              0.4031 |     0.3590 |     0.4433 |
| FM-v2 (re-evaluated) |            16 |              0.4439 |     0.3872 |     0.4873 |
| FM-v2 (re-evaluated) |            32 |              0.4593 |     0.3892 |     0.5156 |
| FM-v2 (re-evaluated) |            43 |              0.7828 |     0.7116 |     0.8298 |
| FM-v2 (re-evaluated) |            64 |              0.4250 |     0.3359 |     0.4916 |
| FM-v2 (re-evaluated) |           128 |              0.4499 |     0.3504 |     0.5212 |
| FM-v2 (re-evaluated) |          1000 |              0.6005 |     0.4956 |     0.6292 |
| Full FM-v3           |             1 |              0.2936 |     0.2553 |     0.3254 |
| Full FM-v3           |             2 |              0.3106 |     0.2695 |     0.3465 |
| Full FM-v3           |             4 |              0.3313 |     0.2752 |     0.3716 |
| Full FM-v3           |             8 |              0.3465 |     0.2793 |     0.3927 |
| Full FM-v3           |            16 |              0.4152 |     0.3265 |     0.4703 |
| Full FM-v3           |            32 |              0.4329 |     0.3388 |     0.4896 |
| Full FM-v3           |            43 |              0.7273 |     0.6287 |     0.8005 |
| Full FM-v3           |            64 |              0.3950 |     0.2982 |     0.4598 |
| Full FM-v3           |           128 |              0.4262 |     0.3098 |     0.4892 |
| Full FM-v3           |          1000 |              0.5615 |     0.3981 |     0.5823 |

### Learning efficiency

AULC integrates balanced accuracy against log2 case budget. Cases-to-threshold is the first nested budget reaching 90% of that run's largest-budget performance.

| variant                         |   log2_case_budget_aulc |   cases_to_90pct_own_max |
|:--------------------------------|------------------------:|-------------------------:|
| Count-neutral local head (γ=1)  |                  0.3633 |                  77.7600 |
| FM-v1-style episodic retraining |                  0.4175 |                 132.6000 |
| FM-v2 (re-evaluated)            |                  0.4259 |                  84.2400 |
| FM-v2 + realistic episodes      |                  0.4245 |                 122.1600 |
| Full FM-v3                      |                  0.3923 |                 123.7200 |
| Full FM-v3, no pretraining      |                  0.3220 |                  58.0000 |
| Global + learned shrinkage      |                  0.3590 |                  72.1600 |
| Global prototypes               |                  0.3613 |                  96.4400 |
| Global–local head               |                  0.3932 |                 129.1600 |
| Local head (learned γ)          |                  0.3898 |                  77.9600 |
| Local head (γ=0)                |                  0.3291 |                 243.0800 |

### Coverage decomposition

`support_pool_availability` estimates P(A), `macro_retrieval_given_pool` estimates P(R|A), and `macro_decision_given_retrieval` estimates P(D|R). `macro_label_recall_at_k` is the unconditional top-k candidate recall, so it also includes support-pool absence.

| variant                         |   case_budget |   support_pool_availability |   macro_label_recall_at_k |   macro_retrieval_given_pool |   macro_decision_given_retrieval |   conditional_balanced_accuracy_pool_covered |   conditional_balanced_accuracy_retrieval_covered |   recall_p10 |   zero_recall_fraction |
|:--------------------------------|--------------:|----------------------------:|--------------------------:|-----------------------------:|---------------------------------:|---------------------------------------------:|--------------------------------------------------:|-------------:|-----------------------:|
| Count-neutral local head (γ=1)  |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.6589 |                                       0.6589 |                                            0.6589 |       0.0360 |                 0.5751 |
| Count-neutral local head (γ=1)  |             2 |                      0.8378 |                    0.4816 |                       0.9995 |                           0.6600 |                                       0.6598 |                                            0.6600 |       0.0458 |                 0.5444 |
| Count-neutral local head (γ=1)  |             4 |                      0.8865 |                    0.5541 |                       0.9899 |                           0.6118 |                                       0.6065 |                                            0.6118 |       0.0600 |                 0.4686 |
| Count-neutral local head (γ=1)  |             8 |                      0.9403 |                    0.6147 |                       0.9468 |                           0.5809 |                                       0.5523 |                                            0.5809 |       0.0510 |                 0.4132 |
| Count-neutral local head (γ=1)  |            16 |                      0.9601 |                    0.6626 |                       0.9215 |                           0.5751 |                                       0.5299 |                                            0.5751 |       0.0649 |                 0.3367 |
| Count-neutral local head (γ=1)  |            32 |                      0.9677 |                    0.7079 |                       0.8947 |                           0.5190 |                                       0.4636 |                                            0.5190 |       0.0635 |                 0.2990 |
| Count-neutral local head (γ=1)  |            43 |                      1.0000 |                    0.9773 |                       0.9773 |                           0.5542 |                                       0.5360 |                                            0.5542 |       0.4250 |                 0.0000 |
| Count-neutral local head (γ=1)  |            64 |                      0.9787 |                    0.7145 |                       0.8377 |                           0.5216 |                                       0.4390 |                                            0.5216 |       0.0416 |                 0.2932 |
| Count-neutral local head (γ=1)  |           128 |                      0.9911 |                    0.7474 |                       0.7979 |                           0.4956 |                                       0.3935 |                                            0.4956 |       0.0344 |                 0.2943 |
| Count-neutral local head (γ=1)  |          1000 |                      1.0000 |                    0.8986 |                       0.8986 |                           0.5195 |                                       0.4691 |                                            0.5195 |       0.2230 |                 0.0769 |
| FM-v1-style episodic retraining |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.6897 |                                       0.6897 |                                            0.6897 |       0.0360 |                 0.5721 |
| FM-v1-style episodic retraining |             2 |                      0.8378 |                    0.4779 |                       0.9936 |                           0.7010 |                                       0.6981 |                                            0.7010 |       0.0486 |                 0.5299 |
| FM-v1-style episodic retraining |             4 |                      0.8865 |                    0.5515 |                       0.9865 |                           0.6762 |                                       0.6714 |                                            0.6762 |       0.0586 |                 0.4832 |
| FM-v1-style episodic retraining |             8 |                      0.9403 |                    0.6126 |                       0.9460 |                           0.6378 |                                       0.6164 |                                            0.6378 |       0.0710 |                 0.4494 |
| FM-v1-style episodic retraining |            16 |                      0.9601 |                    0.6637 |                       0.9230 |                           0.6421 |                                       0.6028 |                                            0.6421 |       0.1070 |                 0.3938 |
| FM-v1-style episodic retraining |            32 |                      0.9677 |                    0.7019 |                       0.8858 |                           0.6455 |                                       0.5767 |                                            0.6455 |       0.1106 |                 0.3579 |
| FM-v1-style episodic retraining |            43 |                      1.0000 |                    0.9861 |                       0.9861 |                           0.7670 |                                       0.7532 |                                            0.7670 |       0.5371 |                 0.0000 |
| FM-v1-style episodic retraining |            64 |                      0.9787 |                    0.7126 |                       0.8354 |                           0.5736 |                                       0.4843 |                                            0.5736 |       0.0043 |                 0.3820 |
| FM-v1-style episodic retraining |           128 |                      0.9911 |                    0.7467 |                       0.7977 |                           0.5912 |                                       0.4793 |                                            0.5912 |       0.0083 |                 0.3427 |
| FM-v1-style episodic retraining |          1000 |                      1.0000 |                    0.9011 |                       0.9011 |                           0.6660 |                                       0.6007 |                                            0.6660 |       0.0600 |                 0.1538 |
| FM-v2 (re-evaluated)            |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.7152 |                                       0.7152 |                                            0.7152 |       0.0373 |                 0.5721 |
| FM-v2 (re-evaluated)            |             2 |                      0.8378 |                    0.4813 |                       0.9991 |                           0.7214 |                                       0.7208 |                                            0.7214 |       0.0493 |                 0.5299 |
| FM-v2 (re-evaluated)            |             4 |                      0.8865 |                    0.5522 |                       0.9873 |                           0.6839 |                                       0.6788 |                                            0.6839 |       0.0592 |                 0.4893 |
| FM-v2 (re-evaluated)            |             8 |                      0.9403 |                    0.6223 |                       0.9574 |                           0.6479 |                                       0.6290 |                                            0.6479 |       0.0906 |                 0.4385 |
| FM-v2 (re-evaluated)            |            16 |                      0.9601 |                    0.6718 |                       0.9346 |                           0.6508 |                                       0.6187 |                                            0.6508 |       0.1054 |                 0.3915 |
| FM-v2 (re-evaluated)            |            32 |                      0.9677 |                    0.7100 |                       0.8950 |                           0.6442 |                                       0.5849 |                                            0.6442 |       0.1067 |                 0.3491 |
| FM-v2 (re-evaluated)            |            43 |                      1.0000 |                    1.0000 |                       1.0000 |                           0.7828 |                                       0.7828 |                                            0.7828 |       0.5061 |                 0.0000 |
| FM-v2 (re-evaluated)            |            64 |                      0.9787 |                    0.7235 |                       0.8475 |                           0.5848 |                                       0.4996 |                                            0.5848 |       0.0035 |                 0.3708 |
| FM-v2 (re-evaluated)            |           128 |                      0.9911 |                    0.7622 |                       0.8137 |                           0.5764 |                                       0.4816 |                                            0.5764 |       0.0083 |                 0.3482 |
| FM-v2 (re-evaluated)            |          1000 |                      1.0000 |                    0.8967 |                       0.8967 |                           0.6675 |                                       0.6005 |                                            0.6675 |       0.0600 |                 0.1538 |
| FM-v2 + realistic episodes      |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.7222 |                                       0.7222 |                                            0.7222 |       0.0360 |                 0.5721 |
| FM-v2 + realistic episodes      |             2 |                      0.8378 |                    0.4814 |                       0.9992 |                           0.7153 |                                       0.7149 |                                            0.7153 |       0.0486 |                 0.5299 |
| FM-v2 + realistic episodes      |             4 |                      0.8865 |                    0.5512 |                       0.9860 |                           0.6882 |                                       0.6817 |                                            0.6882 |       0.0583 |                 0.4832 |
| FM-v2 + realistic episodes      |             8 |                      0.9403 |                    0.6141 |                       0.9469 |                           0.6537 |                                       0.6286 |                                            0.6537 |       0.0875 |                 0.4415 |
| FM-v2 + realistic episodes      |            16 |                      0.9601 |                    0.6652 |                       0.9248 |                           0.6541 |                                       0.6137 |                                            0.6541 |       0.1064 |                 0.3831 |
| FM-v2 + realistic episodes      |            32 |                      0.9677 |                    0.7023 |                       0.8849 |                           0.6482 |                                       0.5797 |                                            0.6482 |       0.1056 |                 0.3558 |
| FM-v2 + realistic episodes      |            43 |                      1.0000 |                    0.9861 |                       0.9861 |                           0.8194 |                                       0.8056 |                                            0.8194 |       0.5167 |                 0.0000 |
| FM-v2 + realistic episodes      |            64 |                      0.9787 |                    0.7153 |                       0.8383 |                           0.5996 |                                       0.5052 |                                            0.5996 |       0.0038 |                 0.3720 |
| FM-v2 + realistic episodes      |           128 |                      0.9911 |                    0.7507 |                       0.8017 |                           0.5845 |                                       0.4789 |                                            0.5845 |       0.0065 |                 0.3513 |
| FM-v2 + realistic episodes      |          1000 |                      1.0000 |                    0.8972 |                       0.8972 |                           0.6716 |                                       0.6041 |                                            0.6716 |       0.0500 |                 0.1538 |
| Full FM-v3                      |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.6761 |                                       0.6761 |                                            0.6761 |       0.0360 |                 0.5751 |
| Full FM-v3                      |             2 |                      0.8378 |                    0.4810 |                       0.9985 |                           0.6582 |                                       0.6573 |                                            0.6582 |       0.0414 |                 0.5460 |
| Full FM-v3                      |             4 |                      0.8865 |                    0.5536 |                       0.9891 |                           0.6108 |                                       0.6073 |                                            0.6108 |       0.0550 |                 0.4997 |
| Full FM-v3                      |             8 |                      0.9403 |                    0.6132 |                       0.9425 |                           0.5730 |                                       0.5507 |                                            0.5730 |       0.0409 |                 0.4271 |
| Full FM-v3                      |            16 |                      0.9601 |                    0.6602 |                       0.9182 |                           0.6121 |                                       0.5754 |                                            0.6121 |       0.0947 |                 0.3407 |
| Full FM-v3                      |            32 |                      0.9677 |                    0.6918 |                       0.8727 |                           0.6108 |                                       0.5510 |                                            0.6108 |       0.1026 |                 0.3226 |
| Full FM-v3                      |            43 |                      1.0000 |                    0.9722 |                       0.9722 |                           0.7518 |                                       0.7273 |                                            0.7518 |       0.4833 |                 0.0000 |
| Full FM-v3                      |            64 |                      0.9787 |                    0.7143 |                       0.8366 |                           0.5417 |                                       0.4636 |                                            0.5417 |       0.0055 |                 0.3230 |
| Full FM-v3                      |           128 |                      0.9911 |                    0.7402 |                       0.7901 |                           0.5585 |                                       0.4558 |                                            0.5585 |       0.0146 |                 0.3052 |
| Full FM-v3                      |          1000 |                      1.0000 |                    0.8993 |                       0.8993 |                           0.6216 |                                       0.5615 |                                            0.6216 |       0.0995 |                 0.0769 |
| Full FM-v3, no pretraining      |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.5726 |                                       0.5726 |                                            0.5726 |       0.0373 |                 0.5782 |
| Full FM-v3, no pretraining      |             2 |                      0.8378 |                    0.4786 |                       0.9945 |                           0.5432 |                                       0.5412 |                                            0.5432 |       0.0347 |                 0.5795 |
| Full FM-v3, no pretraining      |             4 |                      0.8865 |                    0.5512 |                       0.9861 |                           0.4927 |                                       0.4868 |                                            0.4927 |       0.0353 |                 0.5182 |
| Full FM-v3, no pretraining      |             8 |                      0.9403 |                    0.6127 |                       0.9405 |                           0.4777 |                                       0.4542 |                                            0.4777 |       0.0360 |                 0.4683 |
| Full FM-v3, no pretraining      |            16 |                      0.9601 |                    0.6323 |                       0.8747 |                           0.4958 |                                       0.4385 |                                            0.4958 |       0.0496 |                 0.3801 |
| Full FM-v3, no pretraining      |            32 |                      0.9677 |                    0.6723 |                       0.8450 |                           0.5168 |                                       0.4515 |                                            0.5168 |       0.0692 |                 0.3545 |
| Full FM-v3, no pretraining      |            43 |                      1.0000 |                    0.9634 |                       0.9634 |                           0.6539 |                                       0.6212 |                                            0.6539 |       0.2455 |                 0.2500 |
| Full FM-v3, no pretraining      |            64 |                      0.9787 |                    0.6617 |                       0.7747 |                           0.4543 |                                       0.3651 |                                            0.4543 |       0.0019 |                 0.3742 |
| Full FM-v3, no pretraining      |           128 |                      0.9911 |                    0.6958 |                       0.7450 |                           0.4516 |                                       0.3678 |                                            0.4516 |       0.0130 |                 0.3330 |
| Full FM-v3, no pretraining      |          1000 |                      1.0000 |                    0.8256 |                       0.8256 |                           0.4978 |                                       0.4111 |                                            0.4978 |       0.0204 |                 0.1538 |
| Global + learned shrinkage      |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.6500 |                                       0.6500 |                                            0.6500 |       0.0360 |                 0.5763 |
| Global + learned shrinkage      |             2 |                      0.8378 |                    0.4815 |                       0.9994 |                           0.6291 |                                       0.6288 |                                            0.6291 |       0.0372 |                 0.5342 |
| Global + learned shrinkage      |             4 |                      0.8865 |                    0.5527 |                       0.9879 |                           0.5982 |                                       0.5949 |                                            0.5982 |       0.0503 |                 0.4923 |
| Global + learned shrinkage      |             8 |                      0.9403 |                    0.6191 |                       0.9530 |                           0.5415 |                                       0.5211 |                                            0.5415 |       0.0545 |                 0.4448 |
| Global + learned shrinkage      |            16 |                      0.9601 |                    0.6610 |                       0.9205 |                           0.5296 |                                       0.4983 |                                            0.5296 |       0.0540 |                 0.4338 |
| Global + learned shrinkage      |            32 |                      0.9677 |                    0.6975 |                       0.8794 |                           0.5263 |                                       0.4910 |                                            0.5263 |       0.0536 |                 0.3760 |
| Global + learned shrinkage      |            43 |                      1.0000 |                    0.9861 |                       0.9861 |                           0.6318 |                                       0.6212 |                                            0.6318 |       0.2167 |                 0.2500 |
| Global + learned shrinkage      |            64 |                      0.9787 |                    0.7200 |                       0.8430 |                           0.4380 |                                       0.3852 |                                            0.4380 |       0.0017 |                 0.3834 |
| Global + learned shrinkage      |           128 |                      0.9911 |                    0.7549 |                       0.8058 |                           0.4788 |                                       0.4119 |                                            0.4788 |       0.0091 |                 0.3088 |
| Global + learned shrinkage      |          1000 |                      1.0000 |                    0.8987 |                       0.8987 |                           0.5083 |                                       0.4621 |                                            0.5083 |       0.0306 |                 0.0769 |
| Global prototypes               |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.6568 |                                       0.6568 |                                            0.6568 |       0.0360 |                 0.5721 |
| Global prototypes               |             2 |                      0.8378 |                    0.4814 |                       0.9992 |                           0.6438 |                                       0.6434 |                                            0.6438 |       0.0378 |                 0.5269 |
| Global prototypes               |             4 |                      0.8865 |                    0.5527 |                       0.9879 |                           0.6199 |                                       0.6156 |                                            0.6199 |       0.0600 |                 0.4587 |
| Global prototypes               |             8 |                      0.9403 |                    0.6197 |                       0.9535 |                           0.5741 |                                       0.5478 |                                            0.5741 |       0.0478 |                 0.4049 |
| Global prototypes               |            16 |                      0.9601 |                    0.6629 |                       0.9220 |                           0.5485 |                                       0.5130 |                                            0.5485 |       0.0287 |                 0.3967 |
| Global prototypes               |            32 |                      0.9677 |                    0.6991 |                       0.8838 |                           0.5324 |                                       0.4792 |                                            0.5324 |       0.0138 |                 0.3458 |
| Global prototypes               |            43 |                      1.0000 |                    0.9634 |                       0.9634 |                           0.5052 |                                       0.4823 |                                            0.5052 |       0.0500 |                 0.2500 |
| Global prototypes               |            64 |                      0.9787 |                    0.7190 |                       0.8420 |                           0.4949 |                                       0.4235 |                                            0.4949 |       0.0058 |                 0.3252 |
| Global prototypes               |           128 |                      0.9911 |                    0.7516 |                       0.8028 |                           0.4858 |                                       0.4216 |                                            0.4858 |       0.0114 |                 0.2932 |
| Global prototypes               |          1000 |                      1.0000 |                    0.9006 |                       0.9006 |                           0.4847 |                                       0.4407 |                                            0.4847 |       0.0320 |                 0.0769 |
| Global–local head               |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.6730 |                                       0.6730 |                                            0.6730 |       0.0360 |                 0.5751 |
| Global–local head               |             2 |                      0.8378 |                    0.4814 |                       0.9992 |                           0.6676 |                                       0.6672 |                                            0.6676 |       0.0468 |                 0.5434 |
| Global–local head               |             4 |                      0.8865 |                    0.5537 |                       0.9893 |                           0.6068 |                                       0.6030 |                                            0.6068 |       0.0550 |                 0.4966 |
| Global–local head               |             8 |                      0.9403 |                    0.6148 |                       0.9494 |                           0.5716 |                                       0.5570 |                                            0.5716 |       0.0480 |                 0.4250 |
| Global–local head               |            16 |                      0.9601 |                    0.6596 |                       0.9176 |                           0.6074 |                                       0.5676 |                                            0.6074 |       0.0831 |                 0.3507 |
| Global–local head               |            32 |                      0.9677 |                    0.7066 |                       0.8933 |                           0.5915 |                                       0.5493 |                                            0.5915 |       0.0985 |                 0.3063 |
| Global–local head               |            43 |                      1.0000 |                    1.0000 |                       1.0000 |                           0.7273 |                                       0.7273 |                                            0.7273 |       0.4833 |                 0.0000 |
| Global–local head               |            64 |                      0.9787 |                    0.7185 |                       0.8412 |                           0.5286 |                                       0.4584 |                                            0.5286 |       0.0083 |                 0.3280 |
| Global–local head               |           128 |                      0.9911 |                    0.7521 |                       0.8031 |                           0.5444 |                                       0.4602 |                                            0.5444 |       0.0213 |                 0.2961 |
| Global–local head               |          1000 |                      1.0000 |                    0.9005 |                       0.9005 |                           0.6235 |                                       0.5630 |                                            0.6235 |       0.1212 |                 0.0769 |
| Local head (learned γ)          |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.6712 |                                       0.6712 |                                            0.6712 |       0.0360 |                 0.5751 |
| Local head (learned γ)          |             2 |                      0.8378 |                    0.4816 |                       0.9995 |                           0.6626 |                                       0.6624 |                                            0.6626 |       0.0458 |                 0.5413 |
| Local head (learned γ)          |             4 |                      0.8865 |                    0.5539 |                       0.9897 |                           0.6170 |                                       0.6118 |                                            0.6170 |       0.0550 |                 0.4786 |
| Local head (learned γ)          |             8 |                      0.9403 |                    0.6153 |                       0.9484 |                           0.5763 |                                       0.5473 |                                            0.5763 |       0.0387 |                 0.4296 |
| Local head (learned γ)          |            16 |                      0.9601 |                    0.6617 |                       0.9209 |                           0.6195 |                                       0.5731 |                                            0.6195 |       0.0915 |                 0.3427 |
| Local head (learned γ)          |            32 |                      0.9677 |                    0.7079 |                       0.8947 |                           0.5997 |                                       0.5397 |                                            0.5997 |       0.1020 |                 0.3108 |
| Local head (learned γ)          |            43 |                      1.0000 |                    0.9773 |                       0.9773 |                           0.7917 |                                       0.7689 |                                            0.7917 |       0.5000 |                 0.0000 |
| Local head (learned γ)          |            64 |                      0.9787 |                    0.7181 |                       0.8417 |                           0.5398 |                                       0.4558 |                                            0.5398 |       0.0380 |                 0.3292 |
| Local head (learned γ)          |           128 |                      0.9911 |                    0.7478 |                       0.7983 |                           0.5388 |                                       0.4418 |                                            0.5388 |       0.0300 |                 0.3262 |
| Local head (learned γ)          |          1000 |                      1.0000 |                    0.8986 |                       0.8986 |                           0.6225 |                                       0.5599 |                                            0.6225 |       0.0500 |                 0.1538 |
| Local head (γ=0)                |             1 |                      0.7714 |                    0.4310 |                       1.0000 |                           0.6463 |                                       0.6463 |                                            0.6463 |       0.0360 |                 0.6086 |
| Local head (γ=0)                |             2 |                      0.8378 |                    0.4816 |                       0.9995 |                           0.5959 |                                       0.5959 |                                            0.5959 |       0.0347 |                 0.6333 |
| Local head (γ=0)                |             4 |                      0.8865 |                    0.5529 |                       0.9882 |                           0.4994 |                                       0.4977 |                                            0.4994 |       0.0193 |                 0.6250 |
| Local head (γ=0)                |             8 |                      0.9403 |                    0.6181 |                       0.9511 |                           0.4613 |                                       0.4573 |                                            0.4613 |       0.0345 |                 0.5836 |
| Local head (γ=0)                |            16 |                      0.9601 |                    0.6654 |                       0.9258 |                           0.4651 |                                       0.4514 |                                            0.4651 |       0.0491 |                 0.5573 |
| Local head (γ=0)                |            32 |                      0.9677 |                    0.7061 |                       0.8903 |                           0.4793 |                                       0.4450 |                                            0.4793 |       0.0654 |                 0.5125 |
| Local head (γ=0)                |            43 |                      1.0000 |                    0.9773 |                       0.9773 |                           0.6833 |                                       0.6629 |                                            0.6833 |       0.2455 |                 0.2500 |
| Local head (γ=0)                |            64 |                      0.9787 |                    0.7138 |                       0.8373 |                           0.4014 |                                       0.3522 |                                            0.4014 |       0.0000 |                 0.5486 |
| Local head (γ=0)                |           128 |                      0.9911 |                    0.7517 |                       0.8025 |                           0.4050 |                                       0.3476 |                                            0.4050 |       0.0000 |                 0.5225 |
| Local head (γ=0)                |          1000 |                      1.0000 |                    0.8986 |                       0.8986 |                           0.6091 |                                       0.5484 |                                            0.6091 |       0.0250 |                 0.1538 |

Full FM-v3 recall stratified by the number of support prefixes for the true activity:

|   case_budget |      n=0 |      n=1 |    n=2-5 |    n>5 |
|--------------:|---------:|---------:|---------:|-------:|
|        1.0000 |   0.0000 |   0.6973 |   0.2266 | 0.1190 |
|        2.0000 |   0.0000 |   0.5897 |   0.6580 | 0.0437 |
|        4.0000 |   0.0000 |   0.5196 |   0.6310 | 0.0931 |
|        8.0000 |   0.0000 |   0.3708 |   0.4762 | 0.6301 |
|       16.0000 |   0.0000 |   0.4120 |   0.4896 | 0.6313 |
|       32.0000 |   0.0000 |   0.3375 |   0.4444 | 0.6241 |
|       43.0000 | —      | —      | —      | 0.7273 |
|       64.0000 |   0.0000 |   0.3616 |   0.2187 | 0.5850 |
|      128.0000 |   0.0000 |   0.3370 |   0.3858 | 0.5401 |
|     1000.0000 | —      | —      | —      | 0.5615 |

### Calibration and selective prediction

NLL, multiclass Brier score, and ECE assess probability quality. AURC is area under the selective-risk curve; lower is better. Full reliability-bin and risk–coverage coordinates remain in the JSONL artifacts.

| variant                         |    nll |   multiclass_brier |   ece_10 |   aurc |
|:--------------------------------|-------:|-------------------:|---------:|-------:|
| Count-neutral local head (γ=1)  | 3.8592 |             0.6756 |   0.1699 | 0.4053 |
| FM-v1-style episodic retraining | 3.5964 |             0.5074 |   0.1448 | 0.2215 |
| FM-v2 (re-evaluated)            | 3.4907 |             0.4922 |   0.1420 | 0.2083 |
| FM-v2 + realistic episodes      | 3.5290 |             0.4941 |   0.1397 | 0.2120 |
| Full FM-v3                      | 3.8464 |             0.7951 |   0.3263 | 0.4320 |
| Full FM-v3, no pretraining      | 4.0197 |             0.8484 |   0.2603 | 0.5483 |
| Global + learned shrinkage      | 3.8849 |             0.8026 |   0.2856 | 0.4582 |
| Global prototypes               | 3.7900 |             0.7742 |   0.2640 | 0.4559 |
| Global–local head               | 3.8313 |             0.7897 |   0.3158 | 0.4315 |
| Local head (learned γ)          | 3.8672 |             0.6749 |   0.2150 | 0.3499 |
| Local head (γ=0)                | 3.6951 |             0.5896 |   0.1331 | 0.3138 |

### Retrieval and head ablation

| retrieval_mode         |   balanced_accuracy |   accuracy |   macro_f1 |
|:-----------------------|--------------------:|-----------:|-----------:|
| dynamic_expanded_local |              0.3172 |     0.3661 |     0.2515 |
| foundation_knn         |              0.2735 |     0.4532 |     0.2231 |
| global                 |              0.3214 |     0.3849 |     0.2654 |
| global_local           |              0.3532 |     0.4258 |     0.2958 |
| local                  |              0.3640 |     0.4574 |     0.3219 |

### Accuracy–balanced-accuracy prior trade-off

The natural-prior strength β is swept without retraining; β=0 is prior-free and larger β increasingly follows observed support prevalence.

|   prior_strength |   balanced_accuracy |   accuracy |   macro_f1 |
|-----------------:|--------------------:|-----------:|-----------:|
|           0.0000 |              0.3532 |     0.4258 |     0.2958 |
|           0.2500 |              0.3077 |     0.5141 |     0.2721 |
|           0.5000 |              0.2701 |     0.4859 |     0.2303 |
|           0.7500 |              0.2491 |     0.4644 |     0.2043 |
|           1.0000 |              0.2348 |     0.4484 |     0.1889 |
|           1.5000 |              0.2143 |     0.4229 |     0.1646 |

### Natural versus class-aware support acquisition

| support_scenario   |   balanced_accuracy |   accuracy |   support_pool_availability |
|:-------------------|--------------------:|-----------:|----------------------------:|
| class_aware        |              0.3257 |     0.3119 |                      0.9786 |
| natural            |              0.3806 |     0.5398 |                      0.9174 |

## Conventional low-data baselines

Per-log LSTM variants use natural CE, class-weighted CE, logit adjustment, or Balanced Softmax. Weighted logistic regression, balanced random forest, Gaussian Naive Bayes, and TabPFN-v2 use the same fixed handcrafted prefix representation (activity counts, last activity, cost, and time features). For support pools exceeding TabPFN-v2's native ten-class limit, the official many-class extension applies error-correcting output codes. All methods receive the same nested support cases and fixed queries.

| experiment                      |   case_budget |   n_logs |   balanced_accuracy |   accuracy |   macro_f1 |
|:--------------------------------|--------------:|---------:|--------------------:|-----------:|-----------:|
| baseline_gaussian_nb            |             1 |        5 |              0.2410 |     0.4131 |     0.1991 |
| baseline_gaussian_nb            |             2 |        5 |              0.3373 |     0.6377 |     0.2977 |
| baseline_gaussian_nb            |             4 |        5 |              0.3754 |     0.6586 |     0.3386 |
| baseline_gaussian_nb            |             8 |        5 |              0.4038 |     0.6736 |     0.3704 |
| baseline_gaussian_nb            |            16 |        5 |              0.4281 |     0.6548 |     0.3897 |
| baseline_gaussian_nb            |            32 |        5 |              0.4457 |     0.6111 |     0.3881 |
| baseline_gaussian_nb            |            43 |        1 |              0.8056 |     0.8305 |     0.7872 |
| baseline_gaussian_nb            |            64 |        4 |              0.4084 |     0.5292 |     0.3267 |
| baseline_gaussian_nb            |           128 |        4 |              0.4230 |     0.4005 |     0.3274 |
| baseline_gaussian_nb            |          1000 |        1 |              0.4373 |     0.2646 |     0.3910 |
| baseline_lstm_balanced_softmax  |             1 |        5 |              0.3213 |     0.5844 |     0.2775 |
| baseline_lstm_balanced_softmax  |             2 |        5 |              0.3348 |     0.6311 |     0.2951 |
| baseline_lstm_balanced_softmax  |             4 |        5 |              0.3558 |     0.6365 |     0.3122 |
| baseline_lstm_balanced_softmax  |             8 |        5 |              0.3896 |     0.6861 |     0.3463 |
| baseline_lstm_balanced_softmax  |            16 |        5 |              0.4072 |     0.6993 |     0.3744 |
| baseline_lstm_balanced_softmax  |            32 |        5 |              0.4468 |     0.7251 |     0.4098 |
| baseline_lstm_balanced_softmax  |            43 |        1 |              0.7667 |     0.8000 |     0.7182 |
| baseline_lstm_balanced_softmax  |            64 |        4 |              0.4515 |     0.7168 |     0.4007 |
| baseline_lstm_balanced_softmax  |           128 |        4 |              0.5130 |     0.7147 |     0.4594 |
| baseline_lstm_balanced_softmax  |          1000 |        1 |              0.6804 |     0.5587 |     0.5818 |
| baseline_lstm_class_weighted_ce |             1 |        5 |              0.3277 |     0.5728 |     0.2798 |
| baseline_lstm_class_weighted_ce |             2 |        5 |              0.3450 |     0.6138 |     0.2977 |
| baseline_lstm_class_weighted_ce |             4 |        5 |              0.3570 |     0.5866 |     0.3043 |
| baseline_lstm_class_weighted_ce |             8 |        5 |              0.3800 |     0.5695 |     0.3210 |
| baseline_lstm_class_weighted_ce |            16 |        5 |              0.3845 |     0.5701 |     0.3212 |
| baseline_lstm_class_weighted_ce |            32 |        5 |              0.4731 |     0.6443 |     0.4057 |
| baseline_lstm_class_weighted_ce |            43 |        1 |              0.7556 |     0.7966 |     0.6807 |
| baseline_lstm_class_weighted_ce |            64 |        4 |              0.4886 |     0.6763 |     0.4272 |
| baseline_lstm_class_weighted_ce |           128 |        4 |              0.5516 |     0.6909 |     0.4862 |
| baseline_lstm_class_weighted_ce |          1000 |        1 |              0.6815 |     0.6004 |     0.6031 |
| baseline_lstm_logit_adjustment  |             1 |        5 |              0.2664 |     0.4378 |     0.2315 |
| baseline_lstm_logit_adjustment  |             2 |        5 |              0.1308 |     0.1449 |     0.0813 |
| baseline_lstm_logit_adjustment  |             4 |        5 |              0.0545 |     0.0599 |     0.0340 |
| baseline_lstm_logit_adjustment  |             8 |        5 |              0.0386 |     0.0332 |     0.0249 |
| baseline_lstm_logit_adjustment  |            16 |        5 |              0.0896 |     0.1293 |     0.0706 |
| baseline_lstm_logit_adjustment  |            32 |        5 |              0.3105 |     0.4856 |     0.2742 |
| baseline_lstm_logit_adjustment  |            43 |        1 |              0.6861 |     0.7153 |     0.6231 |
| baseline_lstm_logit_adjustment  |            64 |        4 |              0.3824 |     0.6917 |     0.3600 |
| baseline_lstm_logit_adjustment  |           128 |        4 |              0.4923 |     0.7279 |     0.4563 |
| baseline_lstm_logit_adjustment  |          1000 |        1 |              0.6816 |     0.5630 |     0.5857 |
| baseline_lstm_natural_ce        |             1 |        5 |              0.3176 |     0.5843 |     0.2734 |
| baseline_lstm_natural_ce        |             2 |        5 |              0.3237 |     0.6089 |     0.2797 |
| baseline_lstm_natural_ce        |             4 |        5 |              0.3113 |     0.6037 |     0.2670 |
| baseline_lstm_natural_ce        |             8 |        5 |              0.3417 |     0.6574 |     0.3021 |
| baseline_lstm_natural_ce        |            16 |        5 |              0.3476 |     0.6682 |     0.3154 |
| baseline_lstm_natural_ce        |            32 |        5 |              0.3877 |     0.7197 |     0.3592 |
| baseline_lstm_natural_ce        |            43 |        1 |              0.7778 |     0.8136 |     0.7277 |
| baseline_lstm_natural_ce        |            64 |        4 |              0.3454 |     0.7398 |     0.3310 |
| baseline_lstm_natural_ce        |           128 |        4 |              0.4224 |     0.7725 |     0.4046 |
| baseline_lstm_natural_ce        |          1000 |        1 |              0.6217 |     0.6702 |     0.6243 |
| baseline_random_forest          |             1 |        5 |              0.3305 |     0.5793 |     0.2863 |
| baseline_random_forest          |             2 |        5 |              0.3464 |     0.6304 |     0.3083 |
| baseline_random_forest          |             4 |        5 |              0.3807 |     0.6534 |     0.3510 |
| baseline_random_forest          |             8 |        5 |              0.4146 |     0.6947 |     0.3940 |
| baseline_random_forest          |            16 |        5 |              0.4508 |     0.7137 |     0.4287 |
| baseline_random_forest          |            32 |        5 |              0.4720 |     0.7222 |     0.4606 |
| baseline_random_forest          |            43 |        1 |              0.7184 |     0.7288 |     0.7208 |
| baseline_random_forest          |            64 |        4 |              0.4589 |     0.7320 |     0.4476 |
| baseline_random_forest          |           128 |        4 |              0.4962 |     0.7525 |     0.4915 |
| baseline_random_forest          |          1000 |        1 |              0.5994 |     0.6250 |     0.5949 |
| baseline_tabpfn                 |             1 |        5 |              0.3108 |     0.5424 |     0.2683 |
| baseline_tabpfn                 |             2 |        5 |              0.3291 |     0.6029 |     0.2939 |
| baseline_tabpfn                 |             4 |        5 |              0.3860 |     0.6739 |     0.3501 |
| baseline_tabpfn                 |             8 |        5 |              0.4384 |     0.7380 |     0.4112 |
| baseline_tabpfn                 |            16 |        5 |              0.4718 |     0.7538 |     0.4461 |
| baseline_tabpfn                 |            32 |        5 |              0.4909 |     0.7595 |     0.4676 |
| baseline_tabpfn                 |            43 |        1 |              0.8194 |     0.8475 |     0.7953 |
| baseline_tabpfn                 |            64 |        4 |              0.4701 |     0.7746 |     0.4540 |
| baseline_tabpfn                 |           128 |        4 |              0.5149 |     0.7929 |     0.5007 |
| baseline_tabpfn                 |          1000 |        1 |              0.6378 |     0.6787 |     0.6406 |
| baseline_weighted_logistic      |             1 |        5 |              0.2557 |     0.4205 |     0.2112 |
| baseline_weighted_logistic      |             2 |        5 |              0.3098 |     0.5303 |     0.2740 |
| baseline_weighted_logistic      |             4 |        5 |              0.3456 |     0.5654 |     0.3085 |
| baseline_weighted_logistic      |             8 |        5 |              0.3887 |     0.6095 |     0.3537 |
| baseline_weighted_logistic      |            16 |        5 |              0.4319 |     0.6335 |     0.3864 |
| baseline_weighted_logistic      |            32 |        5 |              0.4701 |     0.6527 |     0.4289 |
| baseline_weighted_logistic      |            43 |        1 |              0.7222 |     0.7458 |     0.7006 |
| baseline_weighted_logistic      |            64 |        4 |              0.4748 |     0.6267 |     0.4176 |
| baseline_weighted_logistic      |           128 |        4 |              0.5322 |     0.6186 |     0.4669 |
| baseline_weighted_logistic      |          1000 |        1 |              0.6374 |     0.4762 |     0.5442 |

## Remaining-time results

MAE remains primary. Median absolute error, MAE skill versus the query-set median, D² absolute-error score, R², and empirical interval coverage are supplementary.

| variant                         |   case_budget |   mae_hours |   median_absolute_error_hours |   normalized_mae |   mae_skill_vs_median |   d2_absolute_error |      r2 |   interval_coverage |   mean_interval_width_hours |
|:--------------------------------|--------------:|------------:|------------------------------:|-----------------:|----------------------:|--------------------:|--------:|--------------------:|----------------------------:|
| Count-neutral local head (γ=1)  |             1 |   1760.9916 |                     1435.8058 |           1.1992 |               -0.6504 |             -0.6504 | -1.1295 |              0.2322 |                    757.5030 |
| Count-neutral local head (γ=1)  |             2 |   1444.8728 |                     1075.5019 |           0.9828 |               -0.3647 |             -0.3647 | -0.5303 |              0.6033 |                   3234.9426 |
| Count-neutral local head (γ=1)  |             4 |   1352.8169 |                      940.9169 |           0.9735 |               -0.3245 |             -0.3245 | -0.3780 |              0.7350 |                   4151.1814 |
| Count-neutral local head (γ=1)  |             8 |   1096.0677 |                      732.8168 |           0.8007 |               -0.1024 |             -0.1024 | -0.0064 |              0.8839 |                   5129.2468 |
| Count-neutral local head (γ=1)  |            16 |    968.5103 |                      686.8602 |           0.6994 |                0.0192 |              0.0192 |  0.0819 |              0.9124 |                   4967.2418 |
| Count-neutral local head (γ=1)  |            32 |    987.0909 |                      659.6155 |           0.7058 |                0.0115 |              0.0115 |  0.0653 |              0.9045 |                   4811.3570 |
| Count-neutral local head (γ=1)  |            43 |   3158.6078 |                     1885.3008 |           0.3724 |                0.0984 |              0.0984 |  0.0962 |              0.9153 |                  17766.6474 |
| Count-neutral local head (γ=1)  |            64 |    424.1626 |                      193.6164 |           0.7894 |               -0.0043 |             -0.0043 |  0.0565 |              0.8934 |                   1684.4690 |
| Count-neutral local head (γ=1)  |           128 |    438.2179 |                      197.2050 |           0.7948 |               -0.0109 |             -0.0109 |  0.0626 |              0.9030 |                   1785.4385 |
| Count-neutral local head (γ=1)  |          1000 |    761.4895 |                      333.2747 |           1.0277 |               -0.1556 |             -0.1556 | -0.0791 |              0.9030 |                   2749.9225 |
| FM-v1-style episodic retraining |             1 |   1762.8591 |                     1441.3714 |           1.1982 |               -0.6493 |             -0.6493 | -1.1288 |              0.2320 |                    754.2216 |
| FM-v1-style episodic retraining |             2 |   1454.4159 |                     1077.3097 |           0.9825 |               -0.3617 |             -0.3617 | -0.5308 |              0.6049 |                   3241.0926 |
| FM-v1-style episodic retraining |             4 |   1383.5399 |                      945.7020 |           0.9974 |               -0.3482 |             -0.3482 | -0.4501 |              0.7366 |                   4193.0043 |
| FM-v1-style episodic retraining |             8 |   1109.6640 |                      753.4813 |           0.8054 |               -0.1072 |             -0.1072 | -0.0084 |              0.8882 |                   5135.1842 |
| FM-v1-style episodic retraining |            16 |    980.3673 |                      683.6774 |           0.6991 |                0.0201 |              0.0201 |  0.0844 |              0.9163 |                   5010.7708 |
| FM-v1-style episodic retraining |            32 |    973.4524 |                      662.4236 |           0.7049 |                0.0169 |              0.0169 |  0.0828 |              0.9112 |                   4826.8660 |
| FM-v1-style episodic retraining |            43 |   3040.8146 |                     2047.0275 |           0.3586 |                0.1320 |              0.1320 |  0.1789 |              0.9322 |                  17644.1619 |
| FM-v1-style episodic retraining |            64 |    418.2042 |                      192.5570 |           0.7794 |                0.0073 |              0.0073 |  0.0735 |              0.9023 |                   1685.6508 |
| FM-v1-style episodic retraining |           128 |    432.1488 |                      198.5109 |           0.7873 |               -0.0022 |             -0.0022 |  0.0804 |              0.9037 |                   1786.0277 |
| FM-v1-style episodic retraining |          1000 |    773.4340 |                      327.7200 |           1.0438 |               -0.1738 |             -0.1738 | -0.0953 |              0.8959 |                   2732.3321 |
| FM-v2 (re-evaluated)            |             1 |   1757.4412 |                     1437.5644 |           1.1893 |               -0.6473 |             -0.6473 | -1.1428 |              0.2365 |                    750.6605 |
| FM-v2 (re-evaluated)            |             2 |   1440.1750 |                     1067.1344 |           0.9761 |               -0.3570 |             -0.3570 | -0.5231 |              0.6024 |                   3228.8963 |
| FM-v2 (re-evaluated)            |             4 |   1354.2912 |                      928.1714 |           0.9858 |               -0.3365 |             -0.3365 | -0.4162 |              0.7307 |                   4151.5652 |
| FM-v2 (re-evaluated)            |             8 |   1096.7795 |                      714.8173 |           0.8118 |               -0.1135 |             -0.1135 | -0.0118 |              0.8771 |                   5146.7580 |
| FM-v2 (re-evaluated)            |            16 |    953.4950 |                      635.1151 |           0.6951 |                0.0258 |              0.0258 |  0.0897 |              0.9154 |                   5021.0993 |
| FM-v2 (re-evaluated)            |            32 |    969.7597 |                      623.7209 |           0.7074 |                0.0125 |              0.0125 |  0.0712 |              0.9062 |                   4747.3115 |
| FM-v2 (re-evaluated)            |            43 |   3004.4638 |                     2040.4245 |           0.3543 |                0.1424 |              0.1424 |  0.1782 |              0.9322 |                  17672.5768 |
| FM-v2 (re-evaluated)            |            64 |    424.4509 |                      190.2386 |           0.7911 |               -0.0065 |             -0.0065 |  0.0440 |              0.8941 |                   1688.2402 |
| FM-v2 (re-evaluated)            |           128 |    439.7520 |                      196.8049 |           0.7970 |               -0.0137 |             -0.0137 |  0.0549 |              0.9016 |                   1783.3791 |
| FM-v2 (re-evaluated)            |          1000 |    767.4098 |                      332.1058 |           1.0357 |               -0.1646 |             -0.1646 | -0.0955 |              0.8854 |                   2746.1400 |
| FM-v2 + realistic episodes      |             1 |   1752.4802 |                     1437.5954 |           1.1883 |               -0.6475 |             -0.6475 | -1.1409 |              0.2352 |                    750.1261 |
| FM-v2 + realistic episodes      |             2 |   1431.8135 |                     1066.2722 |           0.9852 |               -0.3645 |             -0.3645 | -0.5338 |              0.6044 |                   3227.4744 |
| FM-v2 + realistic episodes      |             4 |   1345.5292 |                      936.3790 |           0.9960 |               -0.3473 |             -0.3473 | -0.4296 |              0.7309 |                   4141.8490 |
| FM-v2 + realistic episodes      |             8 |   1090.4779 |                      711.2693 |           0.8096 |               -0.1105 |             -0.1105 | -0.0050 |              0.8848 |                   5129.6750 |
| FM-v2 + realistic episodes      |            16 |    951.3891 |                      622.7827 |           0.6958 |                0.0258 |              0.0258 |  0.0908 |              0.9134 |                   4981.1698 |
| FM-v2 + realistic episodes      |            32 |    967.7893 |                      616.9480 |           0.7097 |                0.0104 |              0.0104 |  0.0704 |              0.9074 |                   4723.3917 |
| FM-v2 + realistic episodes      |            43 |   2982.3575 |                     1895.9337 |           0.3517 |                0.1487 |              0.1487 |  0.1875 |              0.9322 |                  17563.0794 |
| FM-v2 + realistic episodes      |            64 |    425.6898 |                      191.4402 |           0.7916 |               -0.0074 |             -0.0074 |  0.0456 |              0.8936 |                   1704.0649 |
| FM-v2 + realistic episodes      |           128 |    441.5944 |                      194.5617 |           0.8001 |               -0.0171 |             -0.0171 |  0.0495 |              0.9001 |                   1781.3148 |
| FM-v2 + realistic episodes      |          1000 |    774.8482 |                      339.4674 |           1.0457 |               -0.1759 |             -0.1759 | -0.0958 |              0.8924 |                   2723.6985 |
| Full FM-v3                      |             1 |   1756.7228 |                     1440.6439 |           1.1943 |               -0.6485 |             -0.6485 | -1.1297 |              0.2312 |                    755.6131 |
| Full FM-v3                      |             2 |   1437.6837 |                     1060.5010 |           0.9759 |               -0.3535 |             -0.3535 | -0.5039 |              0.6049 |                   3228.2056 |
| Full FM-v3                      |             4 |   1358.9791 |                      941.1034 |           0.9871 |               -0.3372 |             -0.3372 | -0.3962 |              0.7329 |                   4161.7935 |
| Full FM-v3                      |             8 |   1090.1071 |                      728.6401 |           0.8007 |               -0.0993 |             -0.0993 |  0.0045 |              0.8868 |                   5120.3137 |
| Full FM-v3                      |            16 |    958.8027 |                      655.3099 |           0.6960 |                0.0257 |              0.0257 |  0.0897 |              0.9156 |                   4989.9382 |
| Full FM-v3                      |            32 |    972.6174 |                      621.2981 |           0.7105 |                0.0095 |              0.0095 |  0.0675 |              0.9071 |                   4805.0590 |
| Full FM-v3                      |            43 |   3022.2742 |                     1894.1272 |           0.3564 |                0.1373 |              0.1373 |  0.1734 |              0.9322 |                  17811.4141 |
| Full FM-v3                      |            64 |    425.4829 |                      191.3354 |           0.7934 |               -0.0100 |             -0.0100 |  0.0471 |              0.8941 |                   1713.5447 |
| Full FM-v3                      |           128 |    440.5078 |                      199.1586 |           0.7972 |               -0.0146 |             -0.0146 |  0.0603 |              0.9046 |                   1814.5119 |
| Full FM-v3                      |          1000 |    773.5492 |                      324.6147 |           1.0439 |               -0.1739 |             -0.1739 | -0.0962 |              0.8942 |                   2718.1862 |
| Full FM-v3, no pretraining      |             1 |   1779.0601 |                     1435.0425 |           1.2138 |               -0.6507 |             -0.6507 | -1.1079 |              0.2289 |                    763.1868 |
| Full FM-v3, no pretraining      |             2 |   1463.3357 |                     1088.5773 |           1.0050 |               -0.3831 |             -0.3831 | -0.5380 |              0.6009 |                   3263.8509 |
| Full FM-v3, no pretraining      |             4 |   1386.6981 |                      902.8648 |           0.9785 |               -0.3299 |             -0.3299 | -0.4804 |              0.7358 |                   4147.5088 |
| Full FM-v3, no pretraining      |             8 |   1125.5705 |                      762.9136 |           0.7989 |               -0.1022 |             -0.1022 | -0.0170 |              0.8826 |                   5053.0980 |
| Full FM-v3, no pretraining      |            16 |   1027.0128 |                      752.7784 |           0.7177 |               -0.0066 |             -0.0066 |  0.0640 |              0.9127 |                   5029.7215 |
| Full FM-v3, no pretraining      |            32 |    991.5459 |                      640.9649 |           0.7175 |                0.0011 |              0.0011 |  0.0763 |              0.9145 |                   4758.7915 |
| Full FM-v3, no pretraining      |            43 |   3116.5290 |                     1896.5710 |           0.3675 |                0.1104 |              0.1104 |  0.1310 |              0.9322 |                  17932.2819 |
| Full FM-v3, no pretraining      |            64 |    419.4024 |                      197.3863 |           0.7852 |                0.0057 |              0.0057 |  0.0889 |              0.9018 |                   1696.2378 |
| Full FM-v3, no pretraining      |           128 |    435.5582 |                      194.5518 |           0.7966 |               -0.0088 |             -0.0088 |  0.0789 |              0.9089 |                   1792.9375 |
| Full FM-v3, no pretraining      |          1000 |    781.6159 |                      334.7952 |           1.0548 |               -0.1862 |             -0.1862 | -0.0905 |              0.9030 |                   2772.5244 |
| Global + learned shrinkage      |             1 |   1765.8300 |                     1442.9041 |           1.2024 |               -0.6517 |             -0.6517 | -1.1281 |              0.2321 |                    756.6488 |
| Global + learned shrinkage      |             2 |   1448.8937 |                     1071.7658 |           0.9860 |               -0.3652 |             -0.3652 | -0.5292 |              0.6037 |                   3235.7335 |
| Global + learned shrinkage      |             4 |   1349.2926 |                      931.3242 |           0.9725 |               -0.3189 |             -0.3189 | -0.3724 |              0.7349 |                   4130.8459 |
| Global + learned shrinkage      |             8 |   1098.5384 |                      720.8085 |           0.7974 |               -0.0964 |             -0.0964 |  0.0024 |              0.8857 |                   5112.3697 |
| Global + learned shrinkage      |            16 |    975.0515 |                      671.3399 |           0.6994 |                0.0200 |              0.0200 |  0.0847 |              0.9148 |                   4989.8965 |
| Global + learned shrinkage      |            32 |    987.0036 |                      647.2525 |           0.7113 |                0.0066 |              0.0066 |  0.0655 |              0.9048 |                   4895.6624 |
| Global + learned shrinkage      |            43 |   3110.3126 |                     1878.2484 |           0.3667 |                0.1122 |              0.1122 |  0.1367 |              0.9322 |                  18278.5281 |
| Global + learned shrinkage      |            64 |    422.7797 |                      192.0340 |           0.7877 |               -0.0023 |             -0.0023 |  0.0592 |              0.8942 |                   1702.4842 |
| Global + learned shrinkage      |           128 |    436.2109 |                      196.1663 |           0.7934 |               -0.0087 |             -0.0087 |  0.0707 |              0.9066 |                   1800.5069 |
| Global + learned shrinkage      |          1000 |    770.8148 |                      333.7377 |           1.0402 |               -0.1698 |             -0.1698 | -0.0901 |              0.8977 |                   2749.7133 |
| Global prototypes               |             1 |   1765.9015 |                     1440.2332 |           1.2030 |               -0.6533 |             -0.6533 | -1.1312 |              0.2303 |                    757.7128 |
| Global prototypes               |             2 |   1445.7325 |                     1071.4640 |           0.9848 |               -0.3632 |             -0.3632 | -0.5263 |              0.6036 |                   3229.2778 |
| Global prototypes               |             4 |   1345.7746 |                      927.9897 |           0.9728 |               -0.3190 |             -0.3190 | -0.3682 |              0.7353 |                   4124.0348 |
| Global prototypes               |             8 |   1095.3275 |                      714.0235 |           0.7976 |               -0.0963 |             -0.0963 |  0.0029 |              0.8864 |                   5107.9547 |
| Global prototypes               |            16 |    970.9573 |                      671.5975 |           0.6990 |                0.0210 |              0.0210 |  0.0856 |              0.9144 |                   5007.8099 |
| Global prototypes               |            32 |    986.0084 |                      667.0497 |           0.7118 |                0.0068 |              0.0068 |  0.0662 |              0.9068 |                   4844.7090 |
| Global prototypes               |            43 |   3090.4564 |                     1906.8730 |           0.3644 |                0.1179 |              0.1179 |  0.1483 |              0.9322 |                  18526.5232 |
| Global prototypes               |            64 |    422.0537 |                      192.6950 |           0.7901 |               -0.0049 |             -0.0049 |  0.0562 |              0.8938 |                   1697.3010 |
| Global prototypes               |           128 |    435.9468 |                      195.3469 |           0.7913 |               -0.0068 |             -0.0068 |  0.0662 |              0.9041 |                   1790.9150 |
| Global prototypes               |          1000 |    767.2029 |                      335.4235 |           1.0354 |               -0.1643 |             -0.1643 | -0.0857 |              0.9048 |                   2750.1871 |
| Global–local head               |             1 |   1764.7271 |                     1442.2608 |           1.1999 |               -0.6523 |             -0.6523 | -1.1344 |              0.2300 |                    755.9212 |
| Global–local head               |             2 |   1442.2871 |                     1075.9650 |           0.9782 |               -0.3572 |             -0.3572 | -0.5089 |              0.6063 |                   3225.2313 |
| Global–local head               |             4 |   1335.1115 |                      920.6546 |           0.9638 |               -0.3097 |             -0.3097 | -0.3545 |              0.7373 |                   4105.1213 |
| Global–local head               |             8 |   1089.2612 |                      713.1685 |           0.7948 |               -0.0933 |             -0.0933 |  0.0079 |              0.8873 |                   5112.4931 |
| Global–local head               |            16 |    960.5919 |                      671.9244 |           0.6948 |                0.0269 |              0.0269 |  0.0914 |              0.9162 |                   4993.8702 |
| Global–local head               |            32 |    969.4552 |                      638.5705 |           0.7099 |                0.0112 |              0.0112 |  0.0749 |              0.9053 |                   4718.5718 |
| Global–local head               |            43 |   3039.5316 |                     1883.7334 |           0.3584 |                0.1324 |              0.1324 |  0.1677 |              0.9322 |                  17058.6872 |
| Global–local head               |            64 |    422.4833 |                      192.3862 |           0.7877 |               -0.0013 |             -0.0013 |  0.0615 |              0.8964 |                   1694.5395 |
| Global–local head               |           128 |    432.2870 |                      194.7165 |           0.7876 |               -0.0014 |             -0.0014 |  0.0847 |              0.9118 |                   1793.3679 |
| Global–local head               |          1000 |    769.5036 |                      332.7362 |           1.0385 |               -0.1678 |             -0.1678 | -0.0856 |              0.8995 |                   2793.6205 |
| Local head (learned γ)          |             1 |   1759.4002 |                     1437.6819 |           1.1984 |               -0.6495 |             -0.6495 | -1.1285 |              0.2336 |                    757.2179 |
| Local head (learned γ)          |             2 |   1443.3400 |                     1073.1747 |           0.9821 |               -0.3635 |             -0.3635 | -0.5282 |              0.6026 |                   3234.6445 |
| Local head (learned γ)          |             4 |   1352.5174 |                      940.1169 |           0.9735 |               -0.3241 |             -0.3241 | -0.3771 |              0.7351 |                   4153.5710 |
| Local head (learned γ)          |             8 |   1095.7742 |                      731.6055 |           0.8010 |               -0.1024 |             -0.1024 | -0.0058 |              0.8856 |                   5132.8325 |
| Local head (learned γ)          |            16 |    968.6115 |                      685.4870 |           0.6987 |                0.0201 |              0.0201 |  0.0827 |              0.9121 |                   4973.3170 |
| Local head (learned γ)          |            32 |    986.5724 |                      672.8891 |           0.7066 |                0.0105 |              0.0105 |  0.0639 |              0.9055 |                   4844.5289 |
| Local head (learned γ)          |            43 |   3159.2768 |                     1886.8817 |           0.3725 |                0.0982 |              0.0982 |  0.1003 |              0.9153 |                  17872.8992 |
| Local head (learned γ)          |            64 |    424.6008 |                      194.2657 |           0.7891 |               -0.0046 |             -0.0046 |  0.0546 |              0.8927 |                   1686.7417 |
| Local head (learned γ)          |           128 |    438.2986 |                      193.5156 |           0.7939 |               -0.0101 |             -0.0101 |  0.0667 |              0.9028 |                   1788.3037 |
| Local head (learned γ)          |          1000 |    764.7971 |                      334.0833 |           1.0321 |               -0.1607 |             -0.1607 | -0.0812 |              0.8959 |                   2740.2210 |
| Local head (γ=0)                |             1 |   1760.5251 |                     1430.2084 |           1.1986 |               -0.6517 |             -0.6517 | -1.1358 |              0.2313 |                    755.8308 |
| Local head (γ=0)                |             2 |   1441.2705 |                     1073.1184 |           0.9849 |               -0.3660 |             -0.3660 | -0.5324 |              0.6024 |                   3233.6761 |
| Local head (γ=0)                |             4 |   1352.6834 |                      940.8992 |           0.9760 |               -0.3276 |             -0.3276 | -0.3893 |              0.7340 |                   4155.3445 |
| Local head (γ=0)                |             8 |   1094.4275 |                      727.7025 |           0.8013 |               -0.1029 |             -0.1029 | -0.0061 |              0.8834 |                   5132.0424 |
| Local head (γ=0)                |            16 |    966.3524 |                      680.2663 |           0.6983 |                0.0203 |              0.0203 |  0.0835 |              0.9116 |                   5004.9790 |
| Local head (γ=0)                |            32 |    981.7381 |                      665.8805 |           0.7058 |                0.0127 |              0.0127 |  0.0701 |              0.9079 |                   4819.8432 |
| Local head (γ=0)                |            43 |   3088.3888 |                     1918.3718 |           0.3642 |                0.1185 |              0.1185 |  0.1446 |              0.9322 |                  18199.6402 |
| Local head (γ=0)                |            64 |    423.6486 |                      193.2620 |           0.7891 |               -0.0041 |             -0.0041 |  0.0575 |              0.8926 |                   1689.0439 |
| Local head (γ=0)                |           128 |    438.0191 |                      195.3513 |           0.7952 |               -0.0119 |             -0.0119 |  0.0642 |              0.9011 |                   1782.9114 |
| Local head (γ=0)                |          1000 |    762.4684 |                      331.4303 |           1.0290 |               -0.1571 |             -0.1571 | -0.0765 |              0.8959 |                   2746.4807 |

## Relationship to published FM-v2 results

The published FM-v2 study reported ordinary accuracy and MAE under percentage-based support fractions. Its reported classification ranges are included only as context:

| log         |   proto_min |   proto_max |   knn_min |   knn_max |
|:------------|------------:|------------:|----------:|----------:|
| billing     |      0.8950 |      0.9450 |    0.9100 |    0.9500 |
| helpdesk    |      0.7250 |      0.8400 |    0.6850 |    0.7800 |
| receipt     |      0.6800 |      0.8500 |    0.6500 |    0.8300 |
| roadtraffic |      0.8500 |      0.9000 |    0.8350 |    0.8850 |
| sepsis      |      0.4460 |      0.5980 |    0.4950 |    0.6400 |

The closest descriptive comparison uses each current run's largest available absolute case budget. Published values remain ranges over percentage-based support fractions, so the table is contextual rather than a matched effect estimate:

| log         |   current_fmv2_case_budget |   current_fmv2_accuracy |   current_fmv2_balanced_accuracy |   current_fmv3_case_budget |   current_fmv3_accuracy |   current_fmv3_balanced_accuracy |   proto_min |   proto_max |   knn_min |   knn_max |
|:------------|---------------------------:|------------------------:|---------------------------------:|---------------------------:|------------------------:|---------------------------------:|------------:|------------:|----------:|----------:|
| billing     |                        128 |                  0.8827 |                           0.4851 |                        128 |                  0.7558 |                           0.4564 |      0.8950 |      0.9450 |    0.9100 |    0.9500 |
| helpdesk    |                        128 |                  0.6662 |                           0.3225 |                        128 |                  0.5462 |                           0.3345 |      0.7250 |      0.8400 |    0.6850 |    0.7800 |
| receipt     |                        128 |                  0.8098 |                           0.4505 |                        128 |                  0.5910 |                           0.3928 |      0.6800 |      0.8500 |    0.6500 |    0.8300 |
| roadtraffic |                         43 |                  0.8136 |                           0.7828 |                         43 |                  0.7458 |                           0.7273 |      0.8500 |      0.9000 |    0.8350 |    0.8850 |
| sepsis      |                       1000 |                  0.5926 |                           0.6005 |                       1000 |                  0.3580 |                           0.5615 |      0.4460 |      0.5980 |    0.4950 |    0.6400 |

Published FM-v2 remaining-time MAE ranges:

| log         |   proto_mae_min |   proto_mae_max |   knn_mae_min |   knn_mae_max |
|:------------|----------------:|----------------:|--------------:|--------------:|
| billing     |       1099.8400 |       1194.6000 |      762.3200 |      862.2400 |
| helpdesk    |          0.1700 |          0.2000 |        0.1500 |        0.1800 |
| receipt     |         94.4700 |        101.7200 |       91.2500 |      100.1100 |
| roadtraffic |       4287.3600 |       4477.5700 |     4468.7400 |     4835.6900 |
| sepsis      |        614.9400 |        788.3000 |      723.4300 |      779.2600 |

They are not paired comparisons: the new experiment uses absolute repeated case budgets and balanced accuracy, and `roadtraffic100traces.xes` has 100 cases rather than the paper's 10,000-case subset. The re-trained `00_fmv2` checkpoint under the new protocol is therefore the authoritative baseline.

Published FM-v1 ranges (historical context only):

| log         |   accuracy_min |   accuracy_max |   mae_min |   mae_max |
|:------------|---------------:|---------------:|----------:|----------:|
| billing     |         0.4200 |         0.5390 | 1089.4800 | 2183.0900 |
| helpdesk    |         0.3180 |         0.4500 |    0.2100 |    0.3300 |
| receipt     |         0.3290 |         0.4280 |   85.1500 |  181.1100 |
| roadtraffic |         0.6160 |         0.6890 | 4933.5800 | 7437.3100 |
| sepsis      |         0.3730 |         0.4830 |  857.7000 | 1291.9800 |

## Answers to the research questions

- **RQ1:** paired balanced accuracy changed by -0.0312, macro-F1 by -0.0655, and ordinary accuracy by -0.1365; full FM-v3 did not retain competitive aggregate predictive performance.
- **RQ2:** at each log's largest evaluated budget, mean pool availability was 0.9929, conditional retrieval P(R|A) was 0.8303, and conditional decision P(D|R) was 0.6038.
- **RQ3:** the best swept β for balanced accuracy was 0; the best β for ordinary accuracy was 0.25.
- **RQ4:** the zero-recall class fraction changed by -0.0184. The global–local mechanism reduced completely ignored classes slightly, but that gain did not translate into higher balanced accuracy.
- **RQ5:** full FM-v3 reached 90% of its own largest-budget balanced accuracy at 123.72 cases on average (nested-budget interpolation was not used).
- **RQ6:** full FM-v3 mean NLL=3.8464, Brier=0.7951, ECE=0.3263, and AURC=0.4320. In particular, ECE=0.3263 does not support a strong calibration claim; risk–coverage coordinates are in JSONL.

## Interpretation and validity

The architecture hypothesis is supported only if improvements in balanced accuracy coincide with reduced zero-recall classes and a smaller gap between pool availability and retrieval coverage. An accuracy gain without those changes is not evidence for the proposed coverage mechanism. Natural-prior rows quantify the ordinary-accuracy operating point; balanced-prior rows quantify the equal-prior operating point. Class-aware support is an acquisition upper bound, not the primary deployment estimate.

The no-pretraining checkpoint separates architectural effects from foundation pretraining. Results on five logs should not be generalized to all enterprise event logs, and the capped largest-case regime on Billing/Helpdesk should be described as a low-data study rather than a full-data benchmark.

## Reproducibility

Raw per-run results are in `evaluation_results/fmv3/<variant>/*.jsonl`; flattened tables and learning-curve summaries are stored alongside them. The external FM-v2 transcription is in `paper_docs/fmv2_paper_reference.csv`. Experiment YAML files are under `configs/fmv3/`.

Sources: [FM-v2 preprint](https://www.alessandroberti.it/new_papers/2026_Berti_FM_Second.pdf), [balanced accuracy definition](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), [Balanced Meta-Softmax](https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html), [official TabPFN repository](https://github.com/PriorLabs/TabPFN).
