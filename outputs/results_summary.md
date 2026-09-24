**Imputation**, test period 2014-06-25 to 2025-12-23 (702 removed temperature values, mean ± std over 3 seed(s)):

- Shared GAN (A): temperature RMSE 0.954 ± 0.066 °C; +42.0% vs linear interpolation
- Shared WGAN-GP (B): temperature RMSE 0.969 ± 0.058 °C; +44.3% vs linear interpolation
- Simple GAN (C1): temperature RMSE 0.904 ± 0.026 °C; +34.6% vs linear interpolation
- Linear interpolation: temperature RMSE 0.672 °C
- Lowest temperature RMSE: **Linear interpolation**

**7-day forecasting, complete context** (596 test windows):

- Persistence: RMSE 1.124 °C
- Simple GAN (C2): RMSE 0.928 ± 0.011 °C; skill vs persistence 0.174 ± 0.010
- Shared GAN (A): RMSE 1.735 ± 0.114 °C; skill vs persistence -0.543 ± 0.102
- Shared WGAN-GP (B): RMSE 1.757 ± 0.233 °C; skill vs persistence -0.563 ± 0.207

**7-day forecasting, gappy context** (596 test windows):

- Persistence: RMSE 1.179 °C
- Simple GAN (C1 → C2): RMSE 0.963 ± 0.016 °C; skill vs persistence 0.183 ± 0.013
- Shared GAN (A): RMSE 1.820 ± 0.099 °C; skill vs persistence -0.544 ± 0.084
- Shared WGAN-GP (B): RMSE 1.836 ± 0.231 °C; skill vs persistence -0.557 ± 0.196

### Imputation, test period

| model | Temp RMSE (°C) ↓ | Temp MAE (°C) ↓ | Temp R² ↑ | All-var RMSE (z) ↓ | All-var MAE (z) ↓ |
|---|---|---|---|---|---|
| Shared GAN (A) | 0.954 ± 0.066 | 0.729 ± 0.045 | 0.848 ± 0.021 | 0.831 ± 0.006 | 0.421 ± 0.001 |
| Shared WGAN-GP (B) | 0.969 ± 0.058 | 0.745 ± 0.041 | 0.843 ± 0.019 | 0.832 ± 0.007 | 0.415 ± 0.007 |
| Simple GAN (C1) | 0.904 ± 0.026 | 0.694 ± 0.015 | 0.864 ± 0.008 | 0.837 ± 0.005 | 0.431 ± 0.008 |
| Linear interpolation | 0.672 | 0.488 | 0.925 | 0.786 | 0.311 |

### 7-day temperature forecasting, test period, complete context

| model | RMSE (°C) ↓ | MAE (°C) ↓ | R² ↑ | Skill vs persistence ↑ |
|---|---|---|---|---|
| Persistence | 1.124 | 0.830 | 0.790 | 0.000 |
| Simple GAN (C2) | 0.928 ± 0.011 | 0.700 ± 0.006 | 0.857 ± 0.003 | 0.174 ± 0.010 |
| Shared GAN (A) | 1.735 ± 0.114 | 1.411 ± 0.109 | 0.499 ± 0.067 | -0.543 ± 0.102 |
| Shared WGAN-GP (B) | 1.757 ± 0.233 | 1.432 ± 0.211 | 0.481 ± 0.135 | -0.563 ± 0.207 |

### 7-day temperature forecasting, test period, gappy context

| model | RMSE (°C) ↓ | MAE (°C) ↓ | R² ↑ | Skill vs persistence ↑ |
|---|---|---|---|---|
| Persistence | 1.179 | 0.865 | 0.769 | 0.000 |
| Simple GAN (C1 → C2) | 0.963 ± 0.016 | 0.731 ± 0.009 | 0.846 ± 0.005 | 0.183 ± 0.013 |
| Shared GAN (A) | 1.820 ± 0.099 | 1.492 ± 0.090 | 0.449 ± 0.060 | -0.544 ± 0.084 |
| Shared WGAN-GP (B) | 1.836 ± 0.231 | 1.505 ± 0.208 | 0.434 ± 0.140 | -0.557 ± 0.196 |
