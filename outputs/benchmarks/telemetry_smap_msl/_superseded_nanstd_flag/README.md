Superseded: `near_zero_spread` was computed with `s_tr.std()`, which is NaN for Chronos-2
(its first 64 context steps have no score), so the flag read False on all 81 channels and the
`nondegenerate` channel subset wrongly included channels that are degenerate for Chronos-2.
Fixed to `np.nanstd`. Only the flag and the subset columns derived from it change; the headline
metrics are computed on all channels.
