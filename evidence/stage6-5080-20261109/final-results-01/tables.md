All planned phases and audits have finished. The full protocol failed its throughput requirements: three original B06 activity-invalid blocks remain excluded. These conditional estimates retain actual pair counts and missing identities, with no activity retry, imputation, repeat reduction or forced completion.

Missing frozen blocks: qwen3_8b_awq / B06 / repeat 0; ministral3_3b / B06 / repeat 0; ministral3_3b / B06 / repeat 3.

Completed offered graph units/s, conditional on the frozen cohort and host. Student t 95% CIs on paired whole-process log throughput ratios; no multiplicity adjustment. Requests within a process are not independent repetitions. Serving seconds exclude startup. Latencies are arithmetic means of process statistics, including process p95s, not pooled request percentiles. Live output lengths, quality, runtimes and interfaces can differ. Throughput is not correctness-adjusted useful work; quality equivalence and universal superiority are not established. Missing external native activity counters are unavailable, not zero.

n shows measured pairs / (measured pairs + reported missing blocks).

| Model / cell | Comparison | n | Numerator mean s | Denominator mean s | Throughput ratio [95% CI] |
| --- | --- | ---: | ---: | ---: | --- |
| qwen3_8b_awq / B01 | Candidate/ARC | 12/12 | 33.457 | 33.502 | 1.0014 [0.9989, 1.0038] |
| qwen3_8b_awq / B02 | Candidate/ARC | 6/6 | 6.220 | 6.268 | 1.0078 [0.9962, 1.0195] |
| qwen3_8b_awq / B03 | Candidate/ARC | 12/12 | 45.641 | 45.512 | 0.9972 [0.9896, 1.0048] |
| qwen3_8b_awq / B04 | Candidate/ARC | 12/12 | 5.781 | 5.830 | 1.0086 [0.9945, 1.0229] |
| qwen3_8b_awq / B05 | Candidate/ARC | 12/12 | 38.105 | 38.227 | 1.0032 [0.9948, 1.0117] |
| qwen3_8b_awq / B06 | Candidate/ARC | 7/8 | 7.742 | 7.572 | 0.9781 [0.9546, 1.0021] |
| qwen3_8b_awq / B06 | Continuum/ARC | 7/8 | 7.337 | 7.572 | 1.0320 [1.0143, 1.0500] |
| qwen3_8b_awq / B06 | Upstream PR #59714/ARC | 7/8 | 7.754 | 7.572 | 0.9766 [0.9501, 1.0039] |
| qwen3_8b_awq / B06 | Candidate/Continuum | 7/8 | 7.742 | 7.337 | 0.9477 [0.9246, 0.9714] |
| qwen3_8b_awq / B06 | Candidate/Upstream PR #59714 | 7/8 | 7.742 | 7.754 | 1.0015 [0.9716, 1.0323] |
| qwen3_8b_awq / B07 | Candidate/ARC | 12/12 | 7.637 | 7.556 | 0.9893 [0.9791, 0.9996] |
| qwen3_8b_awq / B08 | Candidate/ARC | 12/12 | 41.835 | 41.232 | 0.9857 [0.9646, 1.0072] |
| qwen3_8b_awq / B08 | Continuum/ARC | 12/12 | 28.758 | 41.232 | 1.4388 [1.3540, 1.5289] |
| qwen3_8b_awq / B08 | Upstream PR #59714/ARC | 12/12 | 37.685 | 41.232 | 1.0949 [1.0645, 1.1261] |
| qwen3_8b_awq / B08 | Candidate/Continuum | 12/12 | 41.835 | 28.758 | 0.6851 [0.6397, 0.7336] |
| qwen3_8b_awq / B08 | Candidate/Upstream PR #59714 | 12/12 | 41.835 | 37.685 | 0.9003 [0.8690, 0.9327] |
| qwen3_8b_awq / B09 | Candidate/ARC | 12/12 | 43.643 | 42.484 | 0.9733 [0.9617, 0.9850] |
| qwen3_8b_awq / B09 | Continuum/ARC | 12/12 | 29.840 | 42.484 | 1.4258 [1.3702, 1.4837] |
| qwen3_8b_awq / B09 | Upstream PR #59714/ARC | 12/12 | 41.008 | 42.484 | 1.0360 [1.0156, 1.0567] |
| qwen3_8b_awq / B09 | Candidate/Continuum | 12/12 | 43.643 | 29.840 | 0.6826 [0.6558, 0.7106] |
| qwen3_8b_awq / B09 | Candidate/Upstream PR #59714 | 12/12 | 43.643 | 41.008 | 0.9395 [0.9277, 0.9514] |
| qwen3_8b_awq / B10 | Candidate/ARC | 12/12 | 6.065 | 5.976 | 0.9852 [0.9636, 1.0074] |
| qwen3_8b_awq / B11 | Candidate/ARC | 8/8 | 18.662 | 20.028 | 1.0726 [0.9478, 1.2139] |
| qwen3_8b_awq / B11 | Continuum/ARC | 8/8 | 19.905 | 20.028 | 1.0034 [0.8718, 1.1549] |
| qwen3_8b_awq / B11 | Upstream PR #59714/ARC | 8/8 | 18.636 | 20.028 | 1.0711 [0.9761, 1.1753] |
| qwen3_8b_awq / B11 | Candidate/Continuum | 8/8 | 18.662 | 19.905 | 1.0689 [0.9304, 1.2280] |
| qwen3_8b_awq / B11 | Candidate/Upstream PR #59714 | 8/8 | 18.662 | 18.636 | 1.0014 [0.8952, 1.1203] |
| qwen3_8b_awq / B12 | Candidate/ARC | 12/12 | 8.555 | 8.352 | 0.9759 [0.9227, 1.0321] |
| qwen3_8b_awq / B12 | Continuum/ARC | 12/12 | 7.903 | 8.352 | 1.0553 [1.0108, 1.1017] |
| qwen3_8b_awq / B12 | Upstream PR #59714/ARC | 12/12 | 8.432 | 8.352 | 0.9888 [0.9503, 1.0289] |
| qwen3_8b_awq / B12 | Candidate/Continuum | 12/12 | 8.555 | 7.903 | 0.9248 [0.8867, 0.9645] |
| qwen3_8b_awq / B12 | Candidate/Upstream PR #59714 | 12/12 | 8.555 | 8.432 | 0.9869 [0.9504, 1.0248] |
| ministral3_3b / B02 | Candidate/ARC | 6/6 | 6.722 | 6.447 | 0.9613 [0.8980, 1.0290] |
| ministral3_3b / B06 | Candidate/ARC | 4/6 | 25.515 | 25.304 | 0.9905 [0.8376, 1.1711] |
| ministral3_3b / B08 | Candidate/ARC | 6/6 | 31.242 | 31.422 | 1.0053 [0.9705, 1.0413] |
| ministral3_3b / B09 | Candidate/ARC | 6/6 | 31.316 | 31.452 | 1.0043 [0.9926, 1.0161] |
| ministral3_3b / B11 | Candidate/ARC | 6/6 | 25.102 | 25.783 | 1.0257 [0.9567, 1.0997] |
| qwen3_4b_2507 / B02 | Candidate/ARC | 10/10 | 3.836 | 3.836 | 1.0005 [0.9782, 1.0233] |
| qwen3_4b_2507 / B06 | Candidate/ARC | 10/10 | 28.799 | 27.805 | 0.9657 [0.9321, 1.0004] |
| qwen3_4b_2507 / B08 | Candidate/ARC | 10/10 | 48.050 | 43.685 | 0.9107 [0.8740, 0.9489] |
| qwen3_4b_2507 / B09 | Candidate/ARC | 10/10 | 43.697 | 43.344 | 0.9919 [0.9773, 1.0068] |
| qwen3_4b_2507 / B11 | Candidate/ARC | 10/10 | 28.804 | 29.319 | 1.0175 [0.9904, 1.0453] |

Latency values below are milliseconds. Each is the arithmetic mean of the corresponding per-process statistic; a mean of process p95s is not a pooled p95 or a population-tail guarantee.

| Model / cell | Comparison arm | Mean TTFT ms | Mean process p95 TTFT ms | Mean completion ms | Mean process p95 completion ms |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3_8b_awq / B01 | Candidate | 76.241 | 164.217 | 1068.861 | 1521.612 |
| qwen3_8b_awq / B01 | ARC | 81.246 | 187.266 | 1076.579 | 1525.527 |
| qwen3_8b_awq / B02 | Candidate | 2822.843 | 5447.739 | 3137.215 | 5656.632 |
| qwen3_8b_awq / B02 | ARC | 2837.121 | 5463.019 | 3153.108 | 5672.586 |
| qwen3_8b_awq / B03 | Candidate | 20807.810 | 39655.551 | 25531.067 | 44184.825 |
| qwen3_8b_awq / B03 | ARC | 20754.448 | 39535.872 | 25461.058 | 44050.939 |
| qwen3_8b_awq / B04 | Candidate | 625.120 | 1308.778 | 971.127 | 1725.183 |
| qwen3_8b_awq / B04 | ARC | 629.419 | 1317.369 | 978.332 | 1732.137 |
| qwen3_8b_awq / B05 | Candidate | 79.775 | 207.873 | 1120.969 | 1613.316 |
| qwen3_8b_awq / B05 | ARC | 79.660 | 207.722 | 1126.246 | 1625.572 |
| qwen3_8b_awq / B06 | Candidate | 243.923 | 1085.859 | 1638.571 | 2854.829 |
| qwen3_8b_awq / B06 | ARC | 245.478 | 1083.525 | 1622.078 | 2815.143 |
| qwen3_8b_awq / B06 | Continuum | 242.366 | 1090.453 | 1593.470 | 2741.139 |
| qwen3_8b_awq / B06 | Upstream PR #59714 | 241.826 | 1082.029 | 1665.375 | 2858.631 |
| qwen3_8b_awq / B07 | Candidate | 244.322 | 1084.607 | 1620.865 | 2809.993 |
| qwen3_8b_awq / B07 | ARC | 244.040 | 1083.230 | 1618.091 | 2821.959 |
| qwen3_8b_awq / B08 | Candidate | 5282.104 | 13553.726 | 8196.622 | 16643.387 |
| qwen3_8b_awq / B08 | ARC | 5255.203 | 12718.011 | 8040.317 | 15733.894 |
| qwen3_8b_awq / B08 | Continuum | 2563.703 | 14328.352 | 4544.190 | 17041.385 |
| qwen3_8b_awq / B08 | Upstream PR #59714 | 4622.939 | 13290.149 | 7213.762 | 15897.789 |
| qwen3_8b_awq / B09 | Candidate | 5881.409 | 13087.611 | 8820.174 | 15664.495 |
| qwen3_8b_awq / B09 | ARC | 5492.068 | 12631.278 | 8321.630 | 15723.632 |
| qwen3_8b_awq / B09 | Continuum | 2708.248 | 15290.478 | 4770.513 | 18386.012 |
| qwen3_8b_awq / B09 | Upstream PR #59714 | 5085.238 | 13605.931 | 7797.725 | 16533.650 |
| qwen3_8b_awq / B10 | Candidate | 533.099 | 1813.086 | 2072.332 | 3818.802 |
| qwen3_8b_awq / B10 | ARC | 530.140 | 1788.761 | 2041.166 | 3759.956 |
| qwen3_8b_awq / B11 | Candidate | 2156.828 | 9390.120 | 3520.958 | 10493.800 |
| qwen3_8b_awq / B11 | ARC | 2125.788 | 9584.107 | 3463.499 | 10699.474 |
| qwen3_8b_awq / B11 | Continuum | 2240.222 | 10695.025 | 3471.187 | 11501.946 |
| qwen3_8b_awq / B11 | Upstream PR #59714 | 2044.204 | 9505.109 | 3328.432 | 10331.626 |
| qwen3_8b_awq / B12 | Candidate | 258.704 | 1083.924 | 1668.674 | 3004.260 |
| qwen3_8b_awq / B12 | ARC | 259.342 | 1083.456 | 1635.782 | 2869.614 |
| qwen3_8b_awq / B12 | Continuum | 261.977 | 1090.276 | 1557.618 | 2752.413 |
| qwen3_8b_awq / B12 | Upstream PR #59714 | 256.789 | 1081.261 | 1638.269 | 2868.383 |
| ministral3_3b / B02 | Candidate | 2988.132 | 5829.237 | 3352.810 | 6115.598 |
| ministral3_3b / B02 | ARC | 2963.500 | 5623.890 | 3312.610 | 5875.288 |
| ministral3_3b / B06 | Candidate | 1652.511 | 9428.441 | 3970.716 | 11784.934 |
| ministral3_3b / B06 | ARC | 1693.714 | 9570.200 | 4027.390 | 11886.701 |
| ministral3_3b / B08 | Candidate | 3407.208 | 10396.126 | 6337.013 | 13918.682 |
| ministral3_3b / B08 | ARC | 3264.297 | 10601.699 | 6261.153 | 14064.409 |
| ministral3_3b / B09 | Candidate | 3444.413 | 10777.466 | 6401.524 | 14319.987 |
| ministral3_3b / B09 | ARC | 3303.136 | 10665.732 | 6309.040 | 14166.444 |
| ministral3_3b / B11 | Candidate | 2521.448 | 10173.628 | 4519.647 | 11843.786 |
| ministral3_3b / B11 | ARC | 2517.609 | 10106.278 | 4496.023 | 11873.489 |
| qwen3_4b_2507 / B02 | Candidate | 1847.465 | 3447.743 | 2039.283 | 3620.920 |
| qwen3_4b_2507 / B02 | ARC | 1865.650 | 3454.166 | 2057.638 | 3621.489 |
| qwen3_4b_2507 / B06 | Candidate | 2840.489 | 11632.223 | 4982.547 | 13774.010 |
| qwen3_4b_2507 / B06 | ARC | 2830.861 | 11289.425 | 4944.718 | 13372.740 |
| qwen3_4b_2507 / B08 | Candidate | 6894.475 | 14079.301 | 8608.781 | 15738.868 |
| qwen3_4b_2507 / B08 | ARC | 6366.195 | 12057.881 | 7905.869 | 13582.501 |
| qwen3_4b_2507 / B09 | Candidate | 6257.628 | 13311.976 | 7785.254 | 14917.981 |
| qwen3_4b_2507 / B09 | ARC | 6084.545 | 11757.618 | 7557.538 | 13415.799 |
| qwen3_4b_2507 / B11 | Candidate | 3587.105 | 12179.902 | 5348.105 | 13382.196 |
| qwen3_4b_2507 / B11 | ARC | 3613.217 | 11740.089 | 5442.383 | 13033.948 |
