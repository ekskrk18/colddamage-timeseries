# A satellite-based agricultural cold damage index using SMAP Level 4

This repository contains the data processing scripts, processed time-series data, and analysis outputs used in the manuscript:

**A satellite-based agricultural cold damage index integrating subsurface thermal structure and soil moisture using SMAP Level 4**

## Overview

This study analyzes ten agricultural cold damage events in South Korea using NASA SMAP Level 4 data. The Agricultural Cold Damage Index (ACDI) integrates cold intensity, rapid temperature decline, surface–subsurface thermal imbalance, and low-temperature–dryness interaction. A total of 24 ACDI combinations are evaluated using peak dominance, concentration, and threshold exceedance duration within the 48-h pre-event period.

## Data

The original SMAP Level 4 data are publicly available from NASA. This repository provides processed event-based time-series data extracted at the event locations.

Variables include:

- surface temperature
- soil temperature layers 1–6
- surface soil moisture
- root-zone soil moisture
- event time and location
- ACDI metrics and ranking results

## Reproducibility

Run the scripts in the following order:

```bash
python scripts/01_download_SMAP.py
python scripts/02_plot_smap_timeseries.py
python scripts/03_compute_ACDI_timeseries.py
python scripts/04_compute_metrics.py
python scripts/05_rank_ACDI_cases.py
