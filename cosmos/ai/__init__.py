"""COSMOS-AI — learned models layered on top of the classical COSMOS reference code.

This sub-package imports from, but never edits, :mod:`cosmos.health`, :mod:`cosmos.debris`
and :mod:`cosmos.reentry`.  PyTorch is imported lazily, so ``import cosmos`` keeps working
without it.

    data/          acquisition (download), validation, preprocessing, deterministic splits
    telemetry/     ESA-ADB official protocol (forecasters, calibrated alarms, official metrics),
                   OPSSAT-AD classifiers, the learned COSMOS health monitor, robustness
    vision/        SpotGEO (official scorer, heat-map detectors), COSMOS synthetic benchmark
    survival/      leakage-free catalogue landmark: re-entry within a horizon
    surrogate      neural surrogate of the COSMOS re-entry propagator (simulated data)
    uncertainty    conformal p-values, Benjamini-Hochberg, CQR, risk-coverage
    evaluation     metrics, calibration, bootstrap, survival metrics, efficiency
    explain        Integrated Gradients with randomisation and deletion checks
    fusion         evidence-aware decision layer; surrogate footprint for the pipeline
    registry       model cards built from benchmark outputs
    experiment     run tracking (config, seed, environment, checkpoints, metrics)
    report         benchmark aggregation, statistical comparisons, figures
"""
