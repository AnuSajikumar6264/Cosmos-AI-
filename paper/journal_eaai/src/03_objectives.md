# Objectives and problem formulation {#sec:objectives}

## Objectives

1. To build a shared-state digital twin in which orbital debris analysis is the primary function, and lifetime prediction and spacecraft health are sub-functions that feed it.
2. To design a fusion architecture that combines a physics prior with recent tabular neural models and emits calibrated predictive distributions rather than point estimates.
3. To measure what each learned component and each cross-module coupling contributes, against the operational practice it would replace, under forward-in-time evaluation.
4. To report honestly where learning does not help.

## Problem formulation

**Residual lifetime.** For object $i$ with state $s_i(t_0)$ at prediction time $t_0$, let $T_i$ be the time from $t_0$ to natural atmospheric re-entry. $T_i$ is right-censored: an object still in orbit at the catalogue snapshot, or whose phase ended by a controlled de-orbit, landing or docking, contributes only the information $T_i > c_i$ for its censoring time $c_i$. With a five-year horizon the learning target is

$$z_i = \log_{10} \min(T_i,\, 5\ \text{yr}), \qquad \delta_i = \mathbb{1}[T_i \leq c_i],$$ {#eq:target}

and the estimand is the conditional distribution of $z_i$ given $s_i(t_0)$, represented by the quantiles $q_\tau(i)$ for $\tau \in \{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95\}$ and by the horizon probabilities $P(T_i \leq H)$. Models are scored only on objects whose five-year outcome is known at the snapshot, so the five-year cap is a genuine point mass of the target rather than a censoring artefact.

**Collision risk.** For encounter $e$ with the sequence of conjunction data messages $C_e$ released at least two days before closest approach, the target is the risk $r_e$ of the final message, where $r_e = \log_{10} P_c$ is the base-ten logarithm of the collision probability, floored at $-30$. The estimand is the distribution of $r_e$ given $C_e$, and the derived decision quantity is $P(r_e \geq -6)$.

**Health.** For channel $c$ of a spacecraft with nominal history $\mathcal{C}_c$, a forecaster produces a non-conformity score $s_c(t)$, and the estimand is a calibrated alarm: a decision rule whose rate of alarm on nominal data is controlled.

**What integration must mean.** We require each coupling to be a function between module outputs and the shared state, with its inputs restricted to information available at the prediction time. A coupling counts as validated only if removing or replacing it changes a measured quantity on real data by an amount whose confidence interval we report. Architectural connections that cannot be tested on public data are labelled as demonstrations and excluded from every performance claim.
