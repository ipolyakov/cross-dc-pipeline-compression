# Cross-datacenter pipeline compression

## Installation
Build the training image from the Dockerfile. Apply pp_p2p_compression.patch.

## Launch
- Development cluster: ./run_pp.sh NAME ITERS CODEC
- Cross-region: ./run_pp_aws.sh ...

## Tables
| Table | Script | Files |
|-------|--------|-------|
| 3, 4 | analysis/paired_eval.py | results/reeval/ |
| 5, 6 | analysis/sweep_tables.py | results/sweep/ |
| 8 | analysis/wan_table.py | results/aws/ |
| 10 | bench_codec.py | results/microbench/ |

## Not included
Dataset, checkpoints, training-loss curves, calculation of Table 9.

## Note
A seed reproduces a run only on the same GPU model: the GPU random generator gives different initial weights on different models.
