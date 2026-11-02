# Architecture {#sec:arch}

## Shared state

The system keeps one state per tracked object,

$$s_i(t) = \bigl(o_i(t),\, \pi_i,\, h_i(t),\, F_i(t),\, e(t),\, C_i(t)\bigr),$$ {#eq:state}

holding the latest orbit $o_i$ (perigee, apogee, inclination), design properties $\pi_i$ (mass, dimensions, shape and a mass-uncertainty flag), health evidence $h_i$ (catalogue operational status, and an alarm state where telemetry exists), the debris family $F_i$ with sibling statistics restricted to events before $t$, space weather $e$ from observed days only, and a conjunction summary $C_i$ where messages exist. Three modules read and write this state, and every output written to it carries its model, its validation status and its own uncertainty. Figure [-@fig:arch] shows the modules and Figure [-@fig:integration] the couplings.

![**System architecture.** Three modules read and update one state per object. Solid arrows are couplings tested on real data; the dashed coupling has no public test data.](artwork/Figure_4.png){#fig:arch width=92%}

![**Integration.** The four couplings as a data flow, with the variable that crosses each boundary and how each is evaluated. Operational status reaches the re-entry module through the shared state, which is how it is implemented.](artwork_eaai/eaai_integration.png){#fig:integration width=100%}

## The artificial-intelligence contribution

The contribution is a two-expert mixture whose mixing weight is learned from descriptors of *one expert's own reliability*, and whose output is calibrated without distributional assumptions. For object $i$,

$$q_\tau(i) = g(r_i)\, q^{\text{phys}}_\tau(i) + \bigl(1 - g(r_i)\bigr) q^{\text{ml}}_\tau(i), \qquad g(r) = \sigma(w^\top r + b),$$ {#eq:gate}

where $q^{\text{phys}}_\tau$ are the quantiles of a Monte-Carlo physics prior, $q^{\text{ml}}_\tau$ those of a learned expert, and $r_i$ collects reliability descriptors of the physics prior: the provenance of its area-to-mass ratio, the spread of its own Monte-Carlo distribution, eccentricity, a perigee band, the object class, and the probability it places at the five-year cap. Because a convex combination of monotone quantile functions is monotone, Eq. [-@eq:gate] returns a valid predictive distribution for any $g$; no sorting or post-hoc repair is required. The parameters $(w,b)$ minimise the mean pinball loss on out-of-fold predictions of the training period with an $L_2$ penalty, so the gate never sees a target that its experts were fitted on.

Two design choices distinguish this from a standard stacked ensemble. First, the gate input describes the *prior's* trustworthiness, not the input in general, which makes the learned weights interpretable: a positive coefficient means "trust the physics more in this regime". Second, mixing happens in quantile space rather than in expectation, so the fused object remains a distribution and can be conformalised directly.

## Uncertainty layer

Two distinct calibrations are applied, and the distinction matters. **Mondrian conformal prediction** converts the fused quantiles into intervals with finite-sample coverage: on a calibration split disjoint from fitting and gating, the score $S = \max(q_{\text{lo}} - z,\; z - q_{\text{hi}})$ is computed per object, and within group $k$ the correction $\hat{Q}_k$ is the $\lceil (1-\alpha)(n_k+1) \rceil$-th smallest score, giving the interval $[q_{\text{lo}} - \hat{Q}_k,\; q_{\text{hi}} + \hat{Q}_k]$. Groups are object class crossed with perigee band, falling back to class and then to the pooled cohort when a group is too small. **Isotonic regression** is a separate, monotone recalibration of the horizon probabilities $P(T \leq H)$ read off the same fused quantile function. Conformal prediction controls interval coverage and makes no assumption about the score distribution; isotonic regression corrects the probability scale and assumes only monotonicity. They are applied to different outputs and neither substitutes for the other. Because isotonic maps are piecewise constant and therefore tie scores, every ranking metric in this paper is computed on the raw probabilities and every calibration metric on the mapped ones.
