"""COSMOS-AI digital twin: one state per tracked object, three modules, explicit couplings.

state      ObjectTwin: identity, orbit, physical properties, operational/health status,
           conjunction summary, debris family, space weather, and module outputs
couplings  the cross-module links, each with a test on real data where one exists:
             C1 debris -> re-entry   breakup/launch-family evidence (ablation in lifetime.run)
             C2 re-entry -> debris   breakup-cloud survival forecasts against the catalogue
             C3 health -> re-entry   operational status selects the re-entry regime
                                     (prospective, CelesTrak snapshots 2026-08-16 -> 2026-10-27)
             C4 health/debris -> collision response  (demonstration only: no paired data)
engine     builds twins and runs the modules and couplings for a set of objects
"""
