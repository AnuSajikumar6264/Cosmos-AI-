"""Catalogue-wide orbital lifetime and re-entry prediction (the COSMOS-AI re-entry study).

Modules
-------
catalogue   leakage-controlled cohort table from GCAT + CelesTrak SATCAT + space weather
physics     semi-analytic drag decay with NRLMSIS 2.1 densities (physics expert)
models      learned experts (gradient-boosted trees, neural quantile model, survival models)
fusion      reliability-gated fusion of experts and Mondrian conformal calibration
run         experiment driver: splits, training, evaluation, ablations
"""
