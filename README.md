# Adversarial Forecasting: A GAN-Based Framework for Time-Series Imputation and Weather Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Melvin0163/Adversarial-Forecasting-A-GAN-Based-Framework-for-Time-Series-Imputation-and-Weather-Prediction/blob/main/Adversarial_Forecasting_GAN.ipynb)

Generative adversarial networks (GANs) for two related problems in daily weather records:

- **Imputation**: reconstructing values missing from a multivariate record, both isolated days and multi-day gaps.
- **Forecasting**: predicting the next 7 days of air temperature from the recent past.

The study uses 76 years (1950–2025) of ERA5-Land daily reanalysis for Chennai, India. Missing data are simulated realistically, and every model is evaluated on a held-out *future* period against simple baselines. The whole study is one end-to-end notebook: [`Adversarial_Forecasting_GAN.ipynb`](Adversarial_Forecasting_GAN.ipynb).

## Models

| Model | Generator | Adversary | Tasks |
|---|---|---|---|
| **A. Shared GAN** | Shared BiLSTM masked-sequence generator | GAIN-style point-wise discriminator (binary cross-entropy) | imputation; forecasting by masking the final 7 days |
| **B. Shared WGAN-GP** | Same shared generator architecture | Block-level Wasserstein critic with gradient penalty | imputation; forecasting by masking the final 7 days |
| **C. Simple GAN** | One network per task | C1: GAIN-style discriminator · C2: conditional discriminator | C1 imputation only · C2 forecasting only |

"Shared" means two things. Models A and B use the same generator architecture, so the comparison between them isolates the adversarial objective (binary cross-entropy vs. Wasserstein distance). And each trained generator serves both tasks: it fills interior gaps, and it forecasts when the last days of a window are masked. The Simple GAN is the task-specific counterpart.

## Data

Daily aggregates from **ERA5-Land** (`ECMWF/ERA5_LAND/DAILY_AGGR`), extracted with Google Earth Engine at 80.23° E, 13.08° N (Chennai), 1950-01-02 to 2025-12-30 (27,757 days). The extract is included in [`data/era5_chennai_1950_2025.csv`](data/era5_chennai_1950_2025.csv).

| Column | Unit | Daily statistic |
|---|---|---|
| `temp` (2 m air temperature, the target) | °C | mean |
| `surface_pressure` | Pa | mean |
| `total_precipitation_sum` | m | total |
| `dewpoint_temperature_2m` | K | mean |
| `u_component_of_wind_10m`, `v_component_of_wind_10m` | m s⁻¹ | mean |

The notebook also contains the Earth Engine extraction code (Section 1), so the extract can be regenerated with a Google Cloud project that has the Earth Engine API enabled.

## Experimental protocol

1. **Chronological split.** The series is cut into non-overlapping 30-day blocks: 70 % for training (1950–2003), 15 % for validation (2003–2014) and 15 % for the held-out test period (2014–2025).
2. **Scaling.** Each variable is z-scored with training-period statistics only.
3. **Synthetic missingness** (fixed seed, identical for every model). For each variable, 10 % of days are removed at random and about 10 % more as contiguous 3–10-day gaps, about 18.5 % of all values in total. Removed values are kept as ground truth.
4. **Model input.** Zero-filled values plus a missing-value indicator for each 30-day block (30 days × 12 channels).
5. **Training.** Each model keeps the weights of its best validation epoch; the test period is never used for any choice.
6. **Evaluation**, averaged over 3 model seeds (mean ± std):
   - *Imputation:* RMSE, MAE and R² for temperature in °C, and all-variable errors, measured only at the removed entries. Baseline: linear interpolation. Errors are also broken down by gap length.
   - *Forecasting:* identical 7-day targets for every model. RMSE, MAE, R², skill against persistence, and error by lead day. The evaluation is repeated with gaps left in the input history.

## Running the notebook

**Google Colab (recommended).** Click the badge above, choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The data are downloaded from this repository automatically.

**Locally** (Python 3.10+):

```bash
git clone https://github.com/Melvin0163/Adversarial-Forecasting-A-GAN-Based-Framework-for-Time-Series-Imputation-and-Weather-Prediction.git
cd Adversarial-Forecasting-A-GAN-Based-Framework-for-Time-Series-Imputation-and-Weather-Prediction
pip install -r requirements.txt
jupyter lab Adversarial_Forecasting_GAN.ipynb
```

A GPU is strongly recommended. Training all four models for three seeds takes several hours on a CPU. To check that everything runs, set `QUICK_TEST = True` in Section 0 (2 epochs, 1 seed, a few minutes). All settings (seeds, split, missingness, hyperparameters) live in the configuration cell of Section 0.

## Outputs

Running the notebook writes everything to `outputs/`:

| Path | Content |
|---|---|
| `results_summary.md` | headline results and the final tables in Markdown |
| `imputation_metrics_*.csv`, `imputation_temp_rmse_by_gap_length.csv` | imputation scores (per seed and summarised) |
| `forecast_metrics_*.csv`, `forecast_rmse_by_lead_day.csv` | forecasting scores (per seed and summarised) |
| `run_summary.json` | configuration, selected epochs, runtimes, all scores and package versions |
| `figures/` | all figures at 200 dpi |
| `models/` | trained generator weights (not tracked by git) |

## Repository structure

```
├── Adversarial_Forecasting_GAN.ipynb   # complete study: data → models → evaluation → discussion
├── data/
│   └── era5_chennai_1950_2025.csv      # ERA5-Land extract used in the study
├── requirements.txt
└── README.md
```

## Limitations

The notebook discusses these in detail (Section 11). In brief:

- The GAIN-style discriminators see the full observation mask. They can therefore saturate, which leaves checkpoint selection to do much of the work.
- The shared generators are trained only on imputation gaps, so their forecasting scores measure transfer rather than dedicated forecasting skill.
- Missingness is synthetic (random and block-wise). The results come from a single site, with point-error metrics only.

## References

- Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383.
- Gorelick, N. et al. (2017). Google Earth Engine: planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18–27.
- Goodfellow, I. et al. (2014). Generative adversarial nets. *NeurIPS*, 27.
- Yoon, J., Jordon, J. & van der Schaar, M. (2018). GAIN: missing data imputation using generative adversarial nets. *ICML*, PMLR 80, 5689–5698.
- Arjovsky, M., Chintala, S. & Bottou, L. (2017). Wasserstein generative adversarial networks. *ICML*, PMLR 70, 214–223.
- Gulrajani, I. et al. (2017). Improved training of Wasserstein GANs. *NeurIPS*, 30.
- Mirza, M. & Osindero, S. (2014). Conditional generative adversarial nets. arXiv:1411.1784.

The full reference list is in the notebook.

## Data licence and attribution

ERA5-Land is produced by ECMWF for the Copernicus Climate Change Service and distributed under the Licence to Use Copernicus Products. The extract in `data/` contains modified Copernicus Climate Change Service information (1950–2025). Neither the European Commission nor ECMWF is responsible for any use of this information.
