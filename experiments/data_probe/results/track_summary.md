## Success out of 17 (wrong fixes in brackets)

| condition | global, no prior | tight: win_sift | tight: ncc→ecc | tight: cascade | loose: win_sift | loose: ncc→ecc | loose: cascade | coast: win_sift | coast: ncc→ecc | coast: cascade |
|---|---|---|---|---|---|---|---|---|---|---|
| tilt L0 | 17 | 17 | 17 | 17 | 17 | 12 (1✗) | 17 | 17 | 1 | 17 |
| tilt L1 | 17 | 17 | 17 | 17 | 17 | 5 | 17 | 17 | 1 | 17 |
| tilt L2 | 0 | 16 | 14 | 16 | 16 | 3 | 16 | 15 | 2 | 15 |
| blur L0 | 17 | 17 | 17 | 17 | 17 | 9 (1✗) | 17 | 17 | 3 | 17 |
| blur L1 | 14 | 15 | 15 | 16 | 14 | 6 | 14 | 14 | 4 | 15 |
| blur L2 | 0 | 0 | 17 | 17 | 0 | 5 | 5 | 0 | 0 | 0 |
| motion L0 | 17 | 17 | 17 | 17 | 17 | 10 | 17 | 17 | 0 | 17 |
| motion L1 | 17 | 16 | 16 | 17 | 16 | 8 | 17 | 16 | 0 | 17 |
| motion L2 | 0 | 0 | 17 | 17 | 0 | 9 | 9 | 0 | 0 | 0 |
| noise_jpeg L0 | 17 | 17 | 17 | 17 | 17 | 4 | 17 | 17 | 2 | 17 |
| noise_jpeg L1 | 17 | 17 | 17 | 17 | 17 | 3 | 17 | 17 | 2 | 17 |
| noise_jpeg L2 | 17 | 17 | 16 | 17 | 17 | 10 | 17 | 17 | 2 | 17 |
| moire L0 | 17 | 17 | 16 | 17 | 17 | 6 | 17 | 17 | 1 | 17 |
| moire L1 | 17 | 17 | 17 | 17 | 17 | 4 | 17 | 17 | 2 | 17 |
| moire L2 | 17 | 17 | 14 | 17 | 17 | 4 | 17 | 16 | 1 | 17 |
| photo_glare L0 | 17 | 17 | 17 | 17 | 17 | 7 | 17 | 17 | 2 | 17 |
| photo_glare L1 | 17 | 17 | 16 | 17 | 17 | 4 | 17 | 17 | 3 | 17 |
| photo_glare L2 | 17 | 17 | 15 | 17 | 17 | 4 | 17 | 16 | 2 | 17 |
| phone_combo L0 | 17 | 17 | 16 | 17 | 17 | 2 | 17 | 17 | 1 | 17 |
| phone_combo L1 | 3 | 16 | 17 | 17 | 16 | 7 | 17 | 15 | 1 | 15 |
| phone_combo L2 | 0 | 0 | 10 | 10 | 0 | 3 | 3 | 0 | 1 | 1 |
| **all 357** | 272 | 301 | 335 | 348 | 300 | 125 (2✗) | 319 | 296 | 31 | 301 |

## Every method, every prior: success / wrong fixes over all 357 trials, and over the 6 hard conditions

| prior | method | all: success | all: wrong | hard (102): success | hard: wrong | found <5 px, ignoring the accept test (hard) | median ms |
|---|---|---|---|---|---|---|---|
| none | global | 272 | 0 | 17 | 0 | 70 | 244 |
| tight | prior | 58 | 19 | 16 | 7 | 16 | 0 |
| tight | win_sift | 301 | 0 | 47 | 0 | 80 | 18 |
| tight | ncc | 235 | 3 | 63 | 1 | 89 | 4 |
| tight | ecc | 87 | 0 | 21 | 0 | 21 | 4 |
| tight | ncc_ecc | 335 | 0 | 90 | 0 | 90 | 13 |
| tight | cascade win_sift→ncc_ecc→global | 348 | 0 | 93 | 0 | – | – |
| tight | cascade ncc_ecc→win_sift→global | 348 | 0 | 93 | 0 | – | – |
| loose | prior | 12 | 218 | 4 | 67 | 4 | 0 |
| loose | win_sift | 300 | 0 | 46 | 0 | 82 | 24 |
| loose | ncc | 51 | 2 | 11 | 1 | 30 | 7 |
| loose | ecc | 15 | 0 | 3 | 0 | 3 | 4 |
| loose | ncc_ecc | 125 | 2 | 33 | 0 | 35 | 18 |
| loose | cascade win_sift→ncc_ecc→global | 319 | 0 | 64 | 0 | – | – |
| loose | cascade ncc_ecc→win_sift→global | 313 | 2 | 62 | 0 | – | – |
| coast | prior | 3 | 321 | 0 | 92 | 0 | 0 |
| coast | win_sift | 296 | 0 | 44 | 0 | 79 | 35 |
| coast | ncc | 15 | 1 | 4 | 1 | 9 | 10 |
| coast | ecc | 1 | 0 | 0 | 0 | 0 | 4 |
| coast | ncc_ecc | 31 | 0 | 8 | 0 | 8 | 25 |
| coast | cascade win_sift→ncc_ecc→global | 301 | 0 | 46 | 0 | – | – |
| coast | cascade ncc_ecc→win_sift→global | 300 | 0 | 45 | 0 | – | – |

## Accept tests: confidence of correct (<5 px) vs wrong (>20 px) answers, all trials and priors

- global: correct n=325 conf p5/p50 = 10.00/150.00; wrong n=8 conf p50/p95/max = 4.50/5.65/6.00; declared threshold 25; correct kept at it: 272/325
- win_sift: correct n=1006 conf p5/p50 = 11.00/158.00; wrong n=3 conf p50/p95/max = 7.00/7.00/7.00; declared threshold 25; correct kept at it: 897/1006
- ncc: correct n=456 conf p5/p50 = 0.22/0.49; wrong n=479 conf p50/p95/max = 0.19/0.27/0.65; declared threshold 0.4; correct kept at it: 301/456
- ecc: correct n=107 conf p5/p50 = 0.70/0.96; wrong n=61 conf p50/p95/max = 0.18/0.25/0.43; declared threshold 0.5; correct kept at it: 103/107
- ncc_ecc: correct n=499 conf p5/p50 = 0.69/0.96; wrong n=296 conf p50/p95/max = 0.22/0.33/0.57; declared threshold 0.5; correct kept at it: 491/499

## Agreement check: win_sift and ncc_ecc both accepted, but their errors differ by more than 5 px (so the fixes are at least 5 px apart)

- tight: both accepted 288, disagree 0
- loose: both accepted 112, disagree 4
- coast: both accepted 29, disagree 1

## Accuracy when correct (median error, px)

- tight win_sift: median 0.35 px, p90 0.46 px (n=301)
- tight ncc: median 0.76 px, p90 2.27 px (n=235)
- tight ncc_ecc: median 0.36 px, p90 0.45 px (n=335)
- loose win_sift: median 0.36 px, p90 0.50 px (n=300)
- loose ncc: median 1.03 px, p90 2.71 px (n=51)
- loose ncc_ecc: median 0.36 px, p90 0.48 px (n=125)
- coast win_sift: median 0.35 px, p90 0.56 px (n=296)
- coast ncc: median 1.29 px, p90 2.31 px (n=15)
- coast ncc_ecc: median 0.36 px, p90 0.72 px (n=31)
- global: median 0.10 px, p90 0.29 px (n=272)

## Agreement check, case by case: which fix was right

| prior | condition | crop | win_sift: error px / inliers | ncc_ecc: error px / ECC correlation |
|---|---|---|---|---|
| loose | tilt L0 | 6 | 0.51 / 177 | 26.84 / 0.57 |
| loose | blur L0 | 6 | 0.51 / 94 | 26.38 / 0.57 |
| loose | phone_combo L1 | 6 | 0.57 / 41 | 6.38 / 0.55 |
| loose | phone_combo L1 | 30 | 0.21 / 73 | 8.23 / 0.50 |
| coast | phone_combo L1 | 33 | 0.25 / 60 | 6.99 / 0.53 |

## Inlier-threshold sweep: correct (<5 px) / wrong (>20 px) fixes kept at each threshold k

Scored on the same frames, so optimistic; confirm on held-out flights.

| method | trials | k=6 | k=8 | k=10 | k=12 | k=15 | k=20 | k=25 |
|---|---|---|---|---|---|---|---|---|
| global, all | 357 | 324 / 1 | 321 / 0 | 310 / 0 | 302 / 0 | 295 / 0 | 280 / 0 | 272 / 0 |
| global, hard | 102 | 69 / 1 | 66 / 0 | 55 / 0 | 47 / 0 | 40 / 0 | 25 / 0 | 17 / 0 |
| win_sift, all priors pooled | 1071 | 1006 / 2 | 1005 / 0 | 979 / 0 | 951 / 0 | 928 / 0 | 905 / 0 | 897 / 0 |
