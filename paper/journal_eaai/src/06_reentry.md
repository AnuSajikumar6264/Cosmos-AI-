# Re-entry module: physics prior and learned expert {#sec:reentry}

Figure [-@fig:method] shows the whole pipeline, including which cohort each component is fitted on.

![**Experimental design of the re-entry module.** Every component is fitted on data strictly earlier than the cohort it is applied to, and the test cohort enters only at the final step.](artwork_eaai/eaai_method.png){#fig:method width=100%}

## Physics prior

The prior integrates the orbit-averaged drag equations for the semi-major axis $a$ and eccentricity $e$ with a thermospheric density from NRLMSIS 2.1 [@emmert2021nrlmsis] evaluated at perigee, and a ballistic coefficient $B = C_D A/m$. For $e \leq 0.05$ it uses King-Hele's Bessel-function series [@kinghele1987satellite]; above that it integrates the same orbit averages exactly over the eccentric anomaly by 257-node quadrature, because the truncated series raises the perigee for $e$ above about 0.3, which is unphysical. Space weather is held at its value at $t_0$, using only days that ended before $t_0$.

The area-to-mass ratio is the dominant unknown, and it is sampled rather than assumed. Each object draws 32 Monte-Carlo values from one of three sources, which is recorded per object: its design value where the catalogue has one, modelled log-normally with a spread of 0.25 dex; a pool of design values of its class estimated on the training cohort, for intact payloads and rocket stages without one; or the NASA Standard Breakup Model [@johnson2001nasa] with a characteristic length drawn from the training distribution of radar-derived sizes of its class, for fragments and components. Each object has its own random stream seeded from its catalogue identifier, so a draw is reproducible per object. The prior is therefore not free of fitted quantities: its class pools are estimated on the training cohort, and we say so rather than presenting it as purely a priori.

## Learned expert

The expert performs quantile regression of $z$ on 53 features in seven groups: orbit, age at $t_0$, design properties, space weather, family evidence $\varphi_F$, five summaries of the physics prior's Monte-Carlo distribution, and object class. Candidates are models published between 2022 and 2026: TabM, a parameter-efficient deep ensemble of multilayer perceptrons sharing most weights [@gorishniy2025tabm]; and two in-context foundation models, TabICLv2 [@qu2025tabicl; @qu2026tabiclv2] and Kumo-Tabular [@nvidia2026kumo], which condition on the training set at inference time instead of fitting weights to it. The candidate with the lowest pinball loss on the tuning split becomes the expert. Gradient-boosted trees [@chen2016xgboost; @ke2017lightgbm; @prokhorenkova2018catboost], a plain non-crossing quantile multilayer perceptron, and an accelerated-failure-time model trained on all objects including censored ones [@barnwal2022survival] are reported as reference baselines and are not eligible for selection. Each tuned learner receives the same budget of 12 random configurations scored on the tuning split and five seeds; the two in-context models are used with their default inference settings and are not tuned.

The deliberate exclusion is the radar cross-section. Whether the catalogue publishes one depends on how fast the object decayed, so its *availability* carries the outcome; Section [-@sec:results] quantifies this and the paper keeps the covariate only as a labelled leak demonstration.

## State estimation for objects already in orbit

The module is fitted at canonical-orbit epochs, but a twin must forecast objects that have been in orbit for years. Survival since the canonical orbit selects low area-to-mass ratios, which neither the class pools nor the learned expert know about. For each uncontrolled object the twin therefore conditions the area-to-mass distribution on the object's catalogue history, a coarse two-orbit form of the ballistic-coefficient estimation used in practice [@gondelach2017ballistic; @sang2013estimation; @pardini2008impact]:

1. 128 prior samples are propagated from the canonical orbit to the as-of date with the realised trailing space weather;
2. samples that would already have re-entered receive zero weight;
3. the remainder are weighted by a Gaussian likelihood of the current perigee and apogee, with $\sigma = 10\ \text{km} + 0.3\,|\Delta|$ for an observed change $\Delta$ since the canonical orbit;
4. 32 posterior samples are drawn by systematic resampling; if the effective sample size falls below four, survival alone is used, and if no sample survives, the eight longest survivors are.

The physics prior then forecasts from the current orbit with these samples. This step is Bayesian rather than learned, and Section [-@sec:discussion] returns to that point.
