## PBMC-33k cluster 13: privacy (canonical CPU attacker)

| Method | AUC old | AUC new | Adv old [CI] | Adv new [CI] | seeds new | verdict old | verdict new |
|---|---|---|---|---|---|---|---|
| Baseline | 0.791 | 0.801 |  | 0.601 | 1 |  |  |
| Retrain (April, leaky) | 0.578 | 0.579 | 0.156 [0.08, 0.26] | 0.158 [0.08, 0.26] | 5 | pass | pass |
| Retrain (clean) | 0.578 | 0.529 |  | 0.057 [0.04, 0.18] | 5 |  |  |
| Retain-only FT | 0.666 | 0.669 | 0.333 [0.23, 0.43] | 0.339 [0.24, 0.44] | 5 | fail | fail |
| Gradient ascent | 0.698 | 0.706 | 0.396 [0.30, 0.49] | 0.412 [0.31, 0.51] | 5 | fail | fail |
| SSD | 0.718 | 0.701 | 0.435 [0.31, 0.56] | 0.402 [0.28, 0.53] | 3 | fail | fail |
| SCRUB | 0.706 | 0.716 | 0.411 [0.28, 0.54] | 0.431 [0.30, 0.56] | 3 | fail | fail |
| Moon | 0.740 | 0.740 | 0.480 [0.36, 0.60] | 0.480 [0.35, 0.60] | 3 | fail | fail |
| Contrastive | 0.164 | 0.170 | 0.673 [0.57, 0.75] | 0.661 [0.56, 0.75] | 3 | fail | fail |
| Fisher scrubbing | 0.808 | 0.673 | 0.615 [0.45, 0.76] | 0.345 [0.22, 0.47] | 3 | fail | fail |
| Fisher (step ablation) |  | 0.378 |  | 0.243 [0.05, 0.48] | 2 |  | fail |
| Extra-gradient lambda=10 | 0.433 | 0.411 | 0.281 [0.20, 0.37] | 0.225 [0.15, 0.33] | 10 | fail | fail |
| Extra-gradient lambda=1 |  | 0.336 |  | 0.327 [0.20, 0.48] | 10 |  | fail |
| DP-SGD | 0.478 | 0.466 | 0.045 [0.03, 0.18] | 0.068 [0.03, 0.19] | 3 | pass | pass |

Success threshold (retrain advantage nested-CI upper): old 0.258 (leaky retrain), new 0.182 (clean retrain)
Advantage estimator null floor: 0.091 [0.004, 0.253]

## PBMC-33k: utility (held-out, ELBO / KL / marker r / ARI)

| Method | ELBO old | ELBO new | KL old | KL new | marker r old | marker r new | ARI new |
|---|---|---|---|---|---|---|---|
| Baseline | 364.2 | 363.9 | 10.55 | 10.83 | 0.831 | 0.831 | 0.468 |
| Retrain (clean) | 365.6 | 364.2 | 9.69 | 10.97 | 0.829 | 0.831 | 0.481 |
| Retrain (April, leaky) |  | 361.9 |  | 10.86 |  | 0.834 | 0.460 |
| Retain-only FT | 363.2 | 363.0 | 10.59 | 11.00 | 0.832 | 0.832 | 0.451 |
| Gradient ascent | 363.4 | 363.2 | 10.53 | 11.00 | 0.832 | 0.831 | 0.463 |
| SSD | 363.9 | 363.6 | 10.40 | 10.72 | 0.831 | 0.831 | 0.475 |
| SCRUB | 363.8 | 363.5 | 10.41 | 10.72 | 0.832 | 0.831 | 0.461 |
| Moon |  | 364.5 |  | 10.83 |  | 0.831 | 0.468 |
| Contrastive | 364.0 | 363.7 | 10.38 | 10.73 | 0.832 | 0.831 | 0.465 |
| Fisher scrubbing | 489.7 | 363.6 | 0.01 | 10.81 | 0.628 | 0.832 | 0.470 |
| Fisher (step ablation) |  | 367.3 |  | 11.13 |  | 0.830 | 0.492 |
| Extra-gradient lambda=10 | 474.9 | 363.8 | 2.10 | 10.78 | 0.691 | 0.832 | 0.470 |
| Extra-gradient lambda=1 |  | 363.3 |  | 10.82 |  | 0.831 | 0.464 |
| DP-SGD | 403.3 | 402.9 | 8.08 | 8.11 | 0.787 | 0.789 | 0.500 |

## Refit attacker control (clean retrains)

baseline refit CV AUC old 0.798 new 0.797; retrain refit adv old 0.134 [0.05, 0.22] new 0.035 [-0.03, 0.10]

## Tabula Muris (structured, cluster 33)

| Model | AUC old (Jan, NB12 attacker for baseline/retrain) | AUC new (one attacker, July TM baseline) |
|---|---|---|
| Baseline | 0.891 | 0.903 |
| Retrain | 0.944 | 0.943 |
| Extra-gradient | 0.874 | 0.723 |
| Fisher (structured) | 0.946 | 0.944 |
| Fisher (scattered) | 0.568 | 0.408 |

## PBMC cluster 12

| Model | Adv old | Adv new [CI] | AUC new | marker r new |
|---|---|---|---|---|
| Baseline | 0.754 | 0.737 [0.59, 0.86] | 0.869 | 0.831 |
| Retrain (clean) |  | 0.628 [0.53, 0.72] | 0.814 | 0.830 |
| Retrain (April, leaky) | 0.640 | 0.643 [0.54, 0.74] | 0.821 | 0.834 |
| Retain-only FT | 0.727 | 0.734 [0.65, 0.81] | 0.867 | 0.832 |
| Fisher scrubbing | 0.452 | 0.717 [0.62, 0.80] | 0.858 | 0.831 |
| Extra-gradient lambda=10 | 0.027 | 0.330 [0.18, 0.48] | 0.410 | 0.832 |

## Attack suite (PBMC, library size restored; retrain = clean seed 42)

| Attack | Baseline old | Baseline new | Retrain old | Retrain new | EG old | EG new | Fisher old | Fisher new |
|---|---|---|---|---|---|---|---|---|
| loss_recon | 0.438 | 0.560 | 0.438 | 0.075 | 0.438 | 0.399 | 0.438 | 0.375 |
| loss_kl | 0.841 | 0.854 | 0.733 | 0.740 | 0.192 | 0.117 | 0.272 | 0.843 |
| loss_elbo | 0.838 | 0.854 | 0.725 | 0.742 | 0.023 | 0.123 | 0.454 | 0.842 |
| likelihood_ratio | 0.840 | 0.991 | 0.840 | 0.991 | 0.734 | 0.665 | 0.744 | 0.983 |
| knn_latent | 0.986 | 0.982 | 0.971 | 0.962 | 0.897 | 0.683 | 0.927 | 0.980 |

## Frozen critics (re-derived, 3 seeds; old = December single seed)

| Config | AUC old | AUC new | Adv new [CI] | LR new | ELBO new | marker r new |
|---|---|---|---|---|---|---|
| frozen_single_lambda5 | 0.997 | 0.003 | 0.995 [0.99, 1.00] | 0.004 | 363.9 | 0.831 |
| frozen_single_lambda10 | 0.982 | 0.003 | 0.994 [0.99, 1.00] | 0.006 | 363.8 | 0.831 |
| frozen_multi_lambda5 | 0.992 | 0.008 | 0.985 [0.97, 0.99] | 0.006 | 402.9 | 0.832 |
| frozen_multi_lambda10 | 0.992 | 0.008 | 0.985 [0.97, 0.99] | 0.005 | 405.8 | 0.832 |

## Structured size ablation (extra-gradient lambda=10, 3 seeds; per-size clean retrains)

| n | negatives | Baseline AUC (LR) | Retrain AUC, adv (LR) | Extra-gradient AUC, adv [CI] (LR) | old EG AUC |
|---|---|---|---|---|---|
| 10 | 18 | 0.550 (0.911) | 0.452, 0.096 (0.457) | 0.354, 0.293 [0.11, 0.60] (0.189) | 0.263 |
| 50 | 46 | 0.707 (0.850) | 0.574, 0.148 (0.561) | 0.368, 0.264 [0.11, 0.43] (0.143) | 0.503 |
| 100 | 64 | 0.765 (0.824) | 0.670, 0.340 (0.512) | 0.420, 0.160 [0.06, 0.29] (0.150) | 0.611 |

## Scattered size ablation (Fisher, 10 seeds; per-size clean retrains)

| n | negatives | Baseline AUC (LR) | Retrain AUC, adv (LR) | Fisher AUC, adv [CI] (LR) | old Fisher AUC | old baseline |
|---|---|---|---|---|---|---|
| 10 | 99 | 0.569 (0.678) | 0.552, 0.104 (0.616) | 0.544, 0.088 [0.10, 0.26] (0.488) | 0.588 | 0.626 |
| 30 | 281 | 0.592 (0.644) | 0.581, 0.163 (0.481) | 0.587, 0.173 [0.12, 0.24] (0.511) | 0.601 | 0.589 |
| 50 | 449 | 0.575 (0.602) | 0.568, 0.135 (0.511) | 0.570, 0.140 [0.10, 0.19] (0.538) | 0.574 | 0.586 |
| 100 | 818 | 0.551 (0.598) | 0.544, 0.087 (0.471) | 0.547, 0.094 [0.06, 0.13] (0.517) | 0.547 | 0.553 |
| 500 | 2449 | 0.517 (0.627) | 0.509, 0.018 (0.510) | 0.514, 0.029 [0.02, 0.05] (0.592) | 0.516 | 0.524 |

## Phase audit (negative ELBO per cell; MLP and reference-LR AUC)

| Stage | forget | matched | retain | unseen | MLP AUC | LR AUC |
|---|---|---|---|---|---|---|
| baseline | 269.8 | 333.1 | 362.9 | 362.8 | 0.799 | 0.995 |
| fisher_step0.001, after scrub | 5652.0 | 1712.9 | 1254.5 | 1261.6 | 0.015 | 0.018 |
| fisher_step0.001, after finetune | 301.3 | 332.5 | 362.3 | 362.5 | 0.691 | 0.914 |
| fisher_step0.003, after scrub | 66398.6 | 12139.3 | 7115.1 | 7311.6 | 0.183 | 0.017 |
| fisher_step0.003, after finetune | 384.1 | 333.3 | 362.6 | 362.7 | 0.422 | 0.208 |
| fisher_step0.01, after scrub | 216490.8 | 211523.5 | 177978.6 | 175711.8 | 0.492 | 0.435 |
| fisher_step0.01, after finetune | 518.4 | 340.8 | 370.4 | 369.4 | 0.393 | 0.168 |
| gradient_ascent, after ascent | 271.1 | 333.0 | 362.8 | 362.9 | 0.798 | 0.993 |
| gradient_ascent, after finetune | 296.7 | 332.3 | 361.5 | 362.3 | 0.698 | 0.945 |

saved outputs/claims_audit.json
