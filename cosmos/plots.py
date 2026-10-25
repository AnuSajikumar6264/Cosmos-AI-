"""All plotting for the notebook, so notebook cells stay one or two lines long.

Every function takes results produced by the three feature packages and returns a matplotlib
figure.  Nothing here computes physics.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .common import output_dir

# one consistent, print-friendly style for the whole report
BLUE, RED, GREEN, AMBER, GREY = "#1F497D", "#C00000", "#2E7D32", "#B8860B", "#7A7A7A"
LBLUE, VIOLET = "#558ED5", "#7B4EA8"
F1, F2, F3 = GREEN, LBLUE, RED          # one colour per feature

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8,
    "axes.edgecolor": "#999999", "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlecolor": BLUE, "axes.labelsize": 11, "font.size": 11,
    "legend.fontsize": 9.5, "legend.frameon": True, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.autolayout": True,
})


def save(fig, name: str) -> str:
    """Write a figure into ``Final_Project/outputs/figures``."""
    import os
    p = os.path.join(output_dir("figures"), f"{name}.png")
    fig.savefig(p)
    return p


# ============================================================ FEATURE 1 — health
def plot_telemetry(ts, subsystem: str = "Propulsion System"):
    """Raw channels for one subsystem, with the fault onset marked."""
    cols = ts.channel_map[subsystem]
    fig, ax = plt.subplots(len(cols), 1, figsize=(9, 4.2), sharex=True)
    ax = np.atleast_1d(ax)
    t_h = ts.frame.t_s / 3600.0
    for a, c in zip(ax, cols):
        a.plot(t_h, ts.frame[c], color=F1, lw=1.1)
        a.set_ylabel(c, fontsize=9.5)
        if ts.fault_onset_index > 0:
            a.axvline(t_h.iloc[ts.fault_onset_index], color=RED, ls="--", lw=1.6)
    ax[0].set_title(f"Feature 1 — telemetry: {subsystem}"
                    + (f"   (fault injected at the dashed line)" if ts.faults else ""))
    ax[-1].set_xlabel("time [hours]")
    return fig


def plot_health(ts, res):
    """Anomaly score over time, and the final per-subsystem degradation."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    t_h = res.t_s / 3600.0
    ax[0].plot(t_h, res.anomaly, color=F1)
    ax[0].axhline(res.threshold, color=AMBER, ls=":", lw=1.6, label="alarm threshold")
    if ts.fault_onset_index > 0:
        ax[0].axvline(t_h[ts.fault_onset_index], color=RED, ls="--", lw=1.6,
                      label="fault onset")
    if res.detected_index > 0:
        ax[0].axvline(t_h[res.detected_index], color=BLUE, ls="-.", lw=1.6, label="alarm")
    ax[0].set_xlabel("time [hours]"); ax[0].set_ylabel("anomaly score")
    ax[0].set_title(f"Anomaly score  ({res.method})"); ax[0].legend(loc="upper left")

    h = res.health_at(-1)
    names = list(h.values)
    vals = [h[k] for k in names]
    colors = [RED if v > 0.7 else AMBER if v > 0.45 else F1 for v in vals]
    ax[1].barh(range(len(names)), vals, color=colors)
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels([n.replace(" System", "").replace(" Control", "")
                           .replace(" Unit", "") for n in names], fontsize=9)
    ax[1].set_xlim(0, 1); ax[1].set_xlabel("degradation")
    ax[1].set_title("Subsystem health (final)"); ax[1].grid(axis="y", visible=False)
    return fig


def plot_coupling(scen):
    """The coupling experiment: same score, different subsystem, different physics."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    lab = [s.replace(" + ", "+\n").replace(" ", "\n", 1) for s in scen.scenario]
    x = np.arange(len(scen))
    ax[0].bar(x, scen.p_fragmentation, color=F3)
    ax[0].set_title("P(fragmentation)"); ax[0].set_ylabel("probability")
    ax[1].bar(x, scen.breakup_altitude_km, color=F2)
    ax[1].set_title("Break-up altitude [km]"); ax[1].set_ylim(50, 95)
    ax[2].bar(x, scen.bc_sigma_log10, color=VIOLET)
    ax[2].set_title("Footprint width  $\\sigma$ log$_{10}$B")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(lab, fontsize=7.5, rotation=30, ha="right")
        a.grid(axis="x", visible=False)
    fig.suptitle("Feature 1 → 3 coupling: the failure MODE selects the break-up physics",
                 fontsize=12.5, fontweight="bold", color=BLUE, y=1.04)
    return fig


# ============================================================ FEATURE 2 — debris
def plot_catalogue(overview, density):
    """Catalogue composition and the measured debris density profile."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    o = overview.head(4)
    ax[0].bar(o.index, o.still_in_orbit, label="still in orbit", color=F2)
    ax[0].bar(o.index, o.decayed, bottom=o.still_in_orbit, label="already re-entered",
              color=GREY)
    ax[0].set_title("Real catalogue composition"); ax[0].set_ylabel("objects")
    ax[0].legend(); ax[0].grid(axis="x", visible=False)
    ax[1].plot(density.alt_mid_km, density.density_per_km3, "o-", color=F2)
    pk = density.loc[density.density_per_km3.idxmax()]
    ax[1].annotate(f"peak {pk.alt_mid_km:.0f} km", (pk.alt_mid_km, pk.density_per_km3),
                   textcoords="offset points", xytext=(8, 6), color=RED, fontweight="bold")
    ax[1].set_yscale("log"); ax[1].set_xlabel("altitude [km]")
    ax[1].set_ylabel("debris per km$^3$"); ax[1].set_title("Measured spatial density")
    return fig


def plot_frames(scene, det=None):
    """Frame A, frame B and the registered difference, with truth circles."""
    from .debris.detect import register_and_difference
    from .debris.scene import to_uint8
    diff, _ = register_and_difference(scene.frame_a, scene.frame_b)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for a, img, t in zip(ax, (scene.frame_a, scene.frame_b, diff),
                         ("Frame A  (t = 0)", "Frame B  (t = Δt)",
                          "Registered difference")):
        a.imshow(to_uint8(img), cmap="gray", origin="upper")
        a.set_title(t); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for r in scene.truth.itertuples():
        ax[2].add_patch(plt.Circle((r.cx, r.cy), 11, fill=False, color=GREEN, lw=1.6))
        ax[1].add_patch(plt.Circle((r.cx_b, r.cy_b), 11, fill=False, color=GREEN, lw=1.6))
    if det is not None and len(det):
        ax[2].plot(det.frame.cx, det.frame.cy, "x", color=RED, ms=8, mew=2,
                   label="detections")
        ax[2].legend(loc="upper right", fontsize=9)
    fig.suptitle("Feature 2 — stars cancel, movers survive  (green = truth)",
                 fontsize=12.5, fontweight="bold", color=BLUE, y=1.02)
    return fig


def plot_detector_scores(agg, yolo_metrics=None):
    """Precision / recall / F1 for each detector."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    rows = [(r.detector, r.precision, r.recall, r.f1) for r in agg.itertuples()]
    if yolo_metrics:
        rows.append(("learned (TinyYOLO)", yolo_metrics["precision"],
                     yolo_metrics["recall"], yolo_metrics["f1"]))
    x = np.arange(len(rows))
    w = 0.26
    for k, (lab, c) in enumerate((("precision", RED), ("recall", AMBER), ("F1", F1))):
        ax.bar(x + (k - 1) * w, [r[k + 1] for r in rows], w, label=lab, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0].replace(" ", "\n") for r in rows], fontsize=9)
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="x", visible=False)
    ax.set_title("Feature 2 — detector comparison")
    for k, r in enumerate(rows):
        ax.text(k + w, r[3] + 0.03, f"{r[3]:.2f}", ha="center", fontsize=9,
                fontweight="bold", color=F1)
    return fig


def plot_lifetime(profile, validation=None):
    """Orbital lifetime against altitude, with the real measured medians overlaid."""
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.semilogy(profile.alt_km, profile.lifetime_years, color=F2, label="model")
    if validation is not None:
        v = validation.dropna(subset=["measured_median_years"])
        ax.plot(v.altitude_km, v.measured_median_years, "o", color=GREEN, ms=9,
                label="measured (real decay dates)")
        for r in v.itertuples():
            ax.annotate(f"{r.event}\n{r.ratio:.2f}×", (r.altitude_km, r.measured_median_years),
                        textcoords="offset points", xytext=(8, -4), fontsize=8.5, color=GREEN)
    ax.axhline(25.0, color=AMBER, ls=":", label="25-year disposal rule")
    ax.set_xlabel("altitude [km]"); ax.set_ylabel("time to re-enter [years]")
    ax.set_title("Feature 2 — how long debris stays up"); ax.legend()
    return fig


# ============================================================ FEATURE 3 — re-entry
def plot_atmosphere(profile, drag_sweep):
    """Density over 14 decades, and C_D across the three flow regimes."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].semilogx(profile.density, profile.alt_km, color=F3)
    ax[0].axhline(86, color=GREY, ls=":", lw=1.4)
    ax[0].text(1e-12, 90, "86 km: closed form → table", fontsize=8.5, color=GREY)
    ax[0].set_xlabel("density [kg/m$^3$]"); ax[0].set_ylabel("altitude [km]")
    ax[0].set_title("US Standard Atmosphere 1976")
    for c, col in zip([c for c in drag_sweep.columns if c != "mach"], (F3, AMBER, F2)):
        ax[1].plot(drag_sweep.mach, drag_sweep[c], label=c, color=col)
    ax[1].set_xlabel("Mach number"); ax[1].set_ylabel("$C_D$")
    ax[1].set_title("Drag depends on Mach AND Knudsen"); ax[1].legend(fontsize=8.5)
    return fig


def plot_trajectory(tr):
    """One fragment resolved: altitude, speed, heating and deceleration."""
    f = tr.frame()
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ax[0].plot(f.t_s, f.alt_km, color=F3)
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("altitude [km]")
    ax[0].set_title("Descent")
    ax[1].plot(f.speed_ms, f.alt_km, color=F3)
    ax[1].set_xlabel("speed [m/s]"); ax[1].set_ylabel("altitude [km]")
    ax[1].set_title("Speed profile")
    ax[2].plot(f.qdot_kW_m2, f.alt_km, color=RED, label="heat flux [kW/m$^2$]")
    ax[2].plot(f.decel_g * 100, f.alt_km, color=BLUE, label="deceleration [0.01 g]")
    ax[2].set_xlabel("value"); ax[2].set_ylabel("altitude [km]")
    ax[2].set_title("Peak heating and peak load"); ax[2].legend(fontsize=8.5)
    fig.suptitle(f"Feature 3 — {tr.name}:  B = {tr.ballistic_coefficient:.0f} kg/m$^2$, "
                 f"lands {tr.downrange_km:.0f} km downrange",
                 fontsize=12, fontweight="bold", color=BLUE, y=1.05)
    return fig


def plot_ballistic_sweep(sweep):
    """Downrange against ballistic coefficient — why one point is not an answer."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.semilogx(sweep.ballistic_coefficient, sweep.downrange_km, "o-", color=F3, ms=8)
    span = sweep.downrange_km.max() - sweep.downrange_km.min()
    ax.set_xlabel("ballistic coefficient  B = m/($C_D$A)  [kg/m$^2$]")
    ax.set_ylabel("downrange [km]")
    ax.set_title(f"Feature 3 — same entry state, {span:,.0f} km of spread from B alone")
    return fig


def plot_footprint(fp, zone=None):
    """Impact scatter coloured by ballistic coefficient, plus the fitted zone."""
    xy = fp.downrange_crossrange_km()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    s = ax[0].scatter(fp.frame.lon, fp.frame.lat, c=np.log10(np.maximum(fp.frame.bc, 1e-3)),
                      s=7, cmap="viridis")
    ax[0].plot(fp.entry_lon, fp.entry_lat, "*", color=RED, ms=16, label="break-up point")
    ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude")
    ax[0].set_title(f"Impact footprint  ({len(fp.frame)} fragments)"); ax[0].legend()
    fig.colorbar(s, ax=ax[0], label="log$_{10}$ B")

    ax[1].scatter(xy[:, 0], xy[:, 1], s=6, color=GREY, alpha=0.6)
    if zone is not None:
        lo, hi = xy.min(0) - 40, xy.max(0) + 40
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 120), np.linspace(lo[1], hi[1], 120))
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        inside = zone["contains"](pts).reshape(gx.shape)
        ax[1].contour(gx, gy, inside.astype(float), levels=[0.5], colors=[RED], linewidths=2)
        ax[1].plot([], [], color=RED, lw=2, label=f"{zone['kind'].replace('_', ' ')}")
        ax[1].legend()
    ax[1].set_xlabel("east [km]"); ax[1].set_ylabel("north [km]")
    ax[1].set_title("Local frame, with the prediction zone")
    ax[1].set_aspect("equal", adjustable="datalim")
    return fig


def plot_zone_comparison(zc):
    """Measured coverage against the nominal target, and the area each zone costs."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(zc))
    w = 0.26
    series = [("gaussian_ellipse_coverage", "Gaussian ellipse", RED),
              ("conformal_ellipse_coverage", "conformal ellipse", AMBER),
              ("conformal_density_coverage", "conformal density", F1)]
    for k, (col, lab, c) in enumerate(series):
        ax[0].bar(x + (k - 1) * w, zc[col] - zc.nominal, w, label=lab, color=c)
    ax[0].axhline(0, color="black", lw=1.2)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{v:.0%}" for v in zc.nominal])
    ax[0].set_xlabel("nominal coverage"); ax[0].set_ylabel("measured − nominal")
    ax[0].set_title("Below zero = under-covers"); ax[0].legend(fontsize=8.5)
    ax[0].grid(axis="x", visible=False)

    for k, (col, lab, c) in enumerate([("gaussian_ellipse_area_km2", "Gaussian ellipse", RED),
                                       ("conformal_ellipse_area_km2", "conformal ellipse", AMBER),
                                       ("conformal_density_area_km2", "conformal density", F1)]):
        ax[1].bar(x + (k - 1) * w, zc[col], w, label=lab, color=c)
    ax[1].set_yscale("log"); ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{v:.0%}" for v in zc.nominal])
    ax[1].set_xlabel("nominal coverage"); ax[1].set_ylabel("area [km$^2$]")
    ax[1].set_title("Cost of that coverage"); ax[1].grid(axis="x", visible=False)
    return fig


def plot_failure_mode_footprints(results):
    """Overlay the footprints of several failure modes at the same anomaly score."""
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for (name, r), c in zip(results.items(), (GREY, AMBER, VIOLET, RED, F2)):
        ax.scatter(r.footprint.frame.lon, r.footprint.frame.lat, s=5, alpha=0.5,
                   color=c, label=f"{name}  (CEP50 {r.footprint.summary()['cep50_km']:.0f} km)")
    first = next(iter(results.values()))
    ax.plot(first.footprint.entry_lon, first.footprint.entry_lat, "*", color="black", ms=16)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Same anomaly score, different failed subsystem → different footprint")
    ax.legend(fontsize=8.5, loc="best")
    return fig
