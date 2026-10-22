## Measured results (this simulator; synthetic workloads; mean of seeds)

| label | prog | sh | base_major_faults_mean | tok_major_faults_mean | pf_reduction_pct_of_means | base_slowdown_mean | tok_slowdown_mean | sd_reduction_pct_of_means | seeds_token_better_slowdown |
|---|---|---|---|---|---|---|---|---|---|
| gzip / vortex3 | gzip* | 40% | 3,938 | 3,938 | +0% | 2.43 | 2.44 | -1% | 0 |
| gzip / vortex3 | vortex3* | 40% | 229 | 158 | +31% | 1.15 | 1.14 | +1% | 5 |
| bit-r / gcc | bit-r* | 40% | 5,310 | 83 | +98% | 2.57 | 1.27 | +51% | 5 |
| bit-r / gcc | gcc* | 40% | 210 | 5,021 | -2289% | 1.45 | 3.07 | -112% | 0 |
| vortex3 / gcc | vortex3* | 40% | 1,102 | 172 | +84% | 1.62 | 1.25 | +22% | 5 |
| vortex3 / gcc | gcc* | 40% | 3,129 | 2,576 | +18% | 2.77 | 2.28 | +18% | 5 |
| vortex1 / vortex3 | vortex1* | 50% | 649 | 947 | -46% | 2.03 | 2.08 | -2% | 3 |
| vortex1 / vortex3 | vortex3* | 50% | 1,015 | 708 | +30% | 1.96 | 1.88 | +4% | 4 |
| LU-1 / LU-2 | LU-1* | 20% | 25,190 | 1,063 | +96% | 42.41 | 2.79 | +93% | 5 |
| LU-1 / LU-2 | LU-2* | 20% | 25,190 | 4,205 | +83% | 42.41 | 5.35 | +87% | 5 |
