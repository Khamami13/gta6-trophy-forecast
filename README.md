# GTA VI Trophy Week-One Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Khamami13/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Khamami13/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for make.fun's **"Forecast how many GTA VI players earn each
trophy in week one"** bounty: given every Exophase read of GTA VI's PS5
trophy page up to a checkpoint, it forecasts each of the game's 55 trophies'
earn rate at the target as five quantiles (5%, 25%, 50%, 75%, 95%), scored
by pinball loss on the logit scale.

## The model

An earn rate is a share of the players Exophase tracks, so it moves for two
reasons at once: more players earn the trophy, and more (mostly fresh)
players join the tracked set and dilute the share. A flat forecast and a
straight line in log-time both miss the shape that matters over a week:
early trophies climb fast while the launch cohort plays, then level off as
the player base broadens.

So each trophy gets its own saturating adoption curve in log-time,

```
r(t) = R / (1 + exp(-(a + b * u))),   u = ln(1 + hours since launch)
```

fitted on the case's history. The ceiling `R` (the share of tracked players
who will ever earn it) is picked by grid search: for each candidate ceiling,
`logit(r / R)` is regressed on `u` by weighted least squares (weights halve
every 24 h, so recent reads count most), scored by squared error on the
logit(rate) scale, and among ceilings within 2% of the best score the
largest is taken — growth is only capped when the data itself shows the
curve bending over. Reads are first compressed to the distinct rates the
page actually showed (Exophase refreshes in steps), keeping the first and
last read. A trophy whose history shows no movement, or a falling trend
(launch-day dilution, not adoption), falls back to its last rate with the
house baseline's random-walk spread.

The spread is honest, not decorative: the fit's residual scale, plus a term
that grows with how far the curve is extrapolated in log-time, plus a floor
for the refresh jumps the history cannot see. Quantiles are the median
shifted by the normal quantiles times that spread, on the logit scale.

On synthetic final-like cases (Weibull adoption plus a dilution drag, a
different shape from the model, rates refreshed in 6-hour steps), this
scores a pinball loss of 0.14–0.22 against 0.24–0.32 for the flat baseline
and 0.25–0.28 for the log-linear curve baseline. On the practice cases the
history is one flat page, so the program correctly falls back to the last
rate with a wide spread — the practice straddles a refresh no history can
foresee.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory and writes `<id>.out` for each:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`. Node 22,
standard library only, no network. Deterministic: two runs on the same case
write the same bytes (about 0.2 s for both practice cases).
