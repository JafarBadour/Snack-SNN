# SNACK Experiment Report

Sources:

- `experiments/gamlp_snack_benchmark/results`
- `experiments/gpt2_dst_lm_benchmark/results/gpt2_gradual_50_to_90`

---

## 1) GAMLP benchmark ( inference only, as requested)

File: `gamlp_inference_synth_5k.csv` (`_5000x5000`, batch size 1)

| Sparsity | Variant | Latency (ms) | Energy (mJ) | Memory (MB) |
|---|---|---:|---:|---:|
| 0.99 | Dense+Mask | 1.0106 | 50.86 | 309.89 |
| 0.99 | SNACK | 0.2072 | 9.82 | 110.60 |
| 0.97 | Dense+Mask | 1.0136 | 52.68 | 409.89 |
| 0.97 | SNACK | 0.2056 | 10.07 | 114.60 |
| 0.95 | Dense+Mask | 1.0135 | 53.63 | 408.56 |
| 0.95 | SNACK | 0.2077 | 10.14 | 118.60 |
| 0.90 | Dense+Mask | 1.0158 | 52.79 | 408.56 |
| 0.90 | SNACK | 0.2452 | 11.82 | 128.60 |

Average SNACK X-factors ( inference, vs Dense+Mask):

- Latency: **4.68x faster** (`1.0134 / 0.2164`)
- Energy: **5.02x lower** (`52.49 / 10.46`)
- Memory: **3.25x lower** (`384.22 / 118.10`)

---

## 2) GPT-2 DST benchmark (`gpt2_gradual_50_to_90`)

### Training (Table A)

| Variant | Loss | Training time/step (ms) | Energy/step (mJ) | Memory (MB) |
|---|---:|---:|---:|---:|
| Dense | 3.7814 | 39.81 | 6103.20 | 2025.66 |
| Dense+Mask | 6.0689 | 41.51 | 6679.45 | 2252.15 |
| SNACK | 6.7600 | 660.06 | 63302.71 | 1573.06 |

SNACK X-factors (training):

- Memory vs Dense: **1.29x lower** (`2025.66 / 1573.06`)
- Memory vs Dense+Mask: **1.43x lower** (`2252.15 / 1573.06`)
- Latency vs Dense: **16.58x slower** (`660.06 / 39.81`)
- Energy vs Dense: **10.37x higher** (`63302.71 / 6103.20`)

### Inference (Table B)

| Variant | Token latency (ms) | Token energy (mJ) | Perplexity |
|---|---:|---:|---:|
| Dense | 5.0007 | 568.40 | 47.38 |
| Dense+Mask | 5.2683 | 702.40 | 200.92 |
| SNACK | 7.1044 | 742.30 | 229.55 |

SNACK X-factors (inference, vs Dense):

- Latency: **1.42x slower** (`7.104 / 5.001`)
- Energy: **1.31x higher** (`742.30 / 568.40`)
- Perplexity: **4.85x higher** (`229.55 / 47.38`, worse)

### Cross-run real table (many aspects)

The table below uses all currently available real GPT-2 result folders with both `table_a` and `table_b` files.

| Run | Model | Sparsity | Train Loss | Train ms/step | Train mJ/step | Train MB | Inference ms/token | Inference mJ/token | Perplexity |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gpt2_8gb` | Dense | - | 3.7814 | 39.83 | 6122.24 | 2025.66 | 5.0613 | 574.60 | 47.38 |
| `gpt2_8gb` | Dense+Mask | 90% | 6.8523 | 41.51 | 6476.87 | 2252.15 | 5.3057 | 700.58 | 578.90 |
| `gpt2_8gb` | SNACK | 90% | 6.2645 | 119.55 | 13500.75 | 1232.99 | 7.1225 | 705.18 | 627.85 |
| `gpt2_from_scratch` | Dense | - | 7.8829 | 43.12 | 7358.11 | 2043.25 | 5.0735 | 629.16 | 1461.08 |
| `gpt2_from_scratch` | Dense+Mask | 90% | 7.5765 | 44.94 | 7622.21 | 2269.74 | 5.3752 | 738.17 | 2166.54 |
| `gpt2_from_scratch` | SNACK | 90% | 7.7045 | 219.76 | 24283.10 | 1250.57 | 7.1561 | 711.71 | 2205.00 |
| `gpt2_gradual_50_to_90` | Dense | - | 3.7814 | 39.81 | 6103.20 | 2025.66 | 5.0007 | 568.40 | 47.38 |
| `gpt2_gradual_50_to_90` | Dense+Mask | 50->90% | 6.0689 | 41.51 | 6679.45 | 2252.15 | 5.2683 | 702.40 | 200.92 |
| `gpt2_gradual_50_to_90` | SNACK | 50->90% | 6.7600 | 660.06 | 63302.71 | 1573.06 | 7.1044 | 742.30 | 229.55 |
| `gpt2_smoke` | Dense | - | 4.2990 | 55.85 | 2135.49 | 2048.99 | 5.1045 | 703.52 | 32.00 |
| `gpt2_smoke` | Dense+Mask | 90% | 7.7925 | 45.20 | 6123.44 | 2274.14 | 5.3428 | 776.83 | 1832.14 |
| `gpt2_smoke` | SNACK | 90% | 6.9389 | 219.26 | 23950.00 | 1255.27 | 7.0594 | 775.78 | 1761.99 |
| `gpt2_wt103_sparsity_sweep/s0p25/run_2c1dd123d1` | Dense | - | 3.6412 | 39.70 | 6191.14 | 2030.70 | 5.0621 | 575.52 | 47.48 |
| `gpt2_wt103_sparsity_sweep/s0p25/run_2c1dd123d1` | Dense+Mask | 25% | 4.2470 | 41.64 | 6916.40 | 2260.84 | 5.4173 | 761.16 | 46.58 |
| `gpt2_wt103_sparsity_sweep/s0p25/run_2c1dd123d1` | SNACK | 25% | 4.8020 | 2806.85 | 345236.72 | 2507.48 | 11.9075 | 2237.78 | 45.93 |

### SNACK X-factors by run (real)

| Run | SNACK memory vs Dense | SNACK train speed vs Dense | SNACK train energy vs Dense | SNACK inf latency vs Dense | SNACK inf energy vs Dense |
|---|---:|---:|---:|---:|---:|
| `gpt2_8gb` | 1.64x lower | 3.00x slower | 2.21x higher | 1.41x slower | 1.23x higher |
| `gpt2_from_scratch` | 1.63x lower | 5.10x slower | 3.30x higher | 1.41x slower | 1.13x higher |
| `gpt2_gradual_50_to_90` | 1.29x lower | 16.58x slower | 10.37x higher | 1.42x slower | 1.31x higher |
| `gpt2_smoke` | 1.63x lower | 3.93x slower | 11.22x higher | 1.38x slower | 1.10x higher |
| `gpt2_wt103_sparsity_sweep/s0p25/run_2c1dd123d1` | 0.81x (higher, not lower) | 70.69x slower | 55.76x higher | 2.35x slower | 3.89x higher |

---

