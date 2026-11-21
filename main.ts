// gta6-trophy-rarity-forecast-01 — GTA VI trophy rarity forecaster.
//
// Reads one case (/input/<id>.in: every Exophase read of GTA VI's PS5 trophy
// page before the checkpoint) and writes /output/<id>.out: for each of the
// game's trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate
// at the case's target, in percent. Node 22, standard library only, no
// network, fully deterministic.
//
// Model — a saturating adoption curve in log-time, per trophy
// -----------------------------------------------------------
// An earn rate is a share of the players Exophase tracks, so it moves for
// two reasons at once: more players earn the trophy, and more (mostly fresh)
// players join the tracked set and dilute the share. A flat forecast and a
// straight line in log-time both miss the shape that matters over a week:
// early trophies climb fast while the launch cohort plays, then level off as
// the player base broadens and the remaining players are the slower ones.
//
// For each trophy we therefore fit, on the case's history,
//
//   r(t) = R / (1 + exp(-(a + b * u))),   u = ln(1 + hours since launch)
//
// a logistic in log-time with a ceiling R (the share of tracked players who
// will ever earn it). The ceiling is picked by a grid search: for each
// candidate R, logit(r / R) is regressed on u by weighted least squares
// (weights halve every 24 h, so the recent reads — the ones that describe
// the player base as it now is — count most), and the score is the weighted
// squared error on the logit(rate) scale. Among ceilings within 2% of the
// best score we take the largest, so growth is only capped when the data
// itself shows the curve bending over. Reads are first compressed to the
// distinct rates the page actually showed (Exophase refreshes in steps, so
// long stretches repeat one page), keeping the first and last read.
//
// Quantiles — an honest spread, not a point forecast
// --------------------------------------------------
// The median is the fitted curve read at the target. The logit spread is
// the fit's residual scale, plus a term that grows with the distance the
// curve is extrapolated in log-time, plus a floor for the refresh jumps the
// history cannot see (a read shows the rates as of Exophase's last refresh,
// so the answering read can step away from any smooth curve). Quantiles are
// the median shifted by the normal quantiles times that spread, on the logit
// scale, clamped to 0.01..99.99 and rounded to 4 decimals.
//
// Fallback — when a trophy's history shows no movement (fewer than 3
// distinct reads, a flat line, or a falling trend, which at launch is the
// dilution of the first flood of new players and says nothing about the
// week), the forecast is the last read's rate with the house baseline's
// random-walk spread, floored so a trophy that has never moved still gets a
// spread wide enough to admit that it will.
//
// Everything the program knows is in the case file. Two runs on the same
// case write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Launch midnight UTC; the house curve baseline counts hours from here too. */
const T0_MS = Date.parse('2026-11-19T00:00:00Z');
/** The house flat baseline's spread, in logit units per square-root day. */
const FLAT_SIGMA = 0.5;
/** Recency half-life of a read's weight, in hours. */
const WEIGHT_HALF_LIFE_H = 24;
/** Extrapolation uncertainty, in logit units per unit of log-time. */
const EXTRAP_SIGMA_PER_U = 0.5;
/** Floor for the refresh jumps the history cannot see, in logit units. */
const REFRESH_SIGMA = 0.25;
/** Narrowest spread we ever report, in logit units. */
const SIGMA_MIN = 0.6;
/** Widest spread we ever report, in logit units. */
const SIGMA_MAX = 1.6;
/** Fallback spread floor, in logit units. */
const FALLBACK_SIGMA_MIN = 0.75;

interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Array<{ id: number }>;
  history: Array<{ at: string; rates: Array<number | null> }>;
}

interface Pt {
  u: number;
  r: number;
  w: number;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** u = ln(1 + hours from launch midnight to this read). */
const uOf = (iso: string): number =>
  Math.log1p(Math.max(0, (Date.parse(iso) - T0_MS) / 3_600_000));

/**
 * Weighted least squares of y on u. Returns slope/intercept, or null when
 * the u values do not vary.
 */
function wls(pts: Pt[], y: number[]): { a: number; b: number } | null {
  let sw = 0;
  let su = 0;
  let sy = 0;
  for (let j = 0; j < pts.length; j++) {
    sw += pts[j].w;
    su += pts[j].w * pts[j].u;
    sy += pts[j].w * y[j];
  }
  const mu = su / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let j = 0; j < pts.length; j++) {
    const du = pts[j].u - mu;
    sxx += pts[j].w * du * du;
    sxy += pts[j].w * du * (y[j] - my);
  }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx;
  return { a: my - b * mu, b };
}

/**
 * Fit the saturating curve for one trophy. `pts` are the distinct reads
 * (u, rate in percent, weight). Returns the median rate at uTarget and the
 * logit spread, or null when the history gives the curve nothing to fit.
 */
function fitCurve(
  pts: Pt[],
  uTarget: number,
): { median: number; sigma: number } | null {
  if (pts.length < 3) return null;
  // Clamp exactly as the scorer does: 0% and 100% have no logit.
  pts = pts.map((p) => ({ ...p, r: clampPct(p.r) }));
  const rMin = Math.min(...pts.map((p) => p.r));
  const rMax = Math.max(...pts.map((p) => p.r));
  if (rMax - rMin < 0.05) return null; // flat line: nothing to fit
  const uLast = pts[pts.length - 1].u;
  if (uTarget <= uLast) return null;
  const uSpan = pts[pts.length - 1].u - pts[0].u;
  if (uSpan < 0.1) return null;

  // Candidate ceilings: multiples of the highest rate seen, up to ~100%.
  // A ceiling must sit above every rate observed, else logit(r / R) is void.
  const factors = [1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2, 2.5, 3, 4, 6, 10, 20];
  const ceilings = [...new Set(factors.map((f) => Math.min(99.9, rMax * f)))].filter(
    (r) => r > rMax,
  );
  if (ceilings.length === 0) return null; // e.g. a trophy stuck at 100%

  let best: { r: number; a: number; b: number; sse: number } | null = null;
  let bestSse = Infinity;
  for (const r of ceilings) {
    const y = pts.map((p) => {
      const f = p.r / r;
      return Math.log(f / (1 - f));
    });
    const fit = wls(pts, y);
    if (fit === null || fit.b < 0) continue; // a falling curve is dilution, not adoption
    let sse = 0;
    let sw = 0;
    for (let j = 0; j < pts.length; j++) {
      const pred = r * sigmoid(fit.a + fit.b * pts[j].u);
      const err = logit(pred) - logit(pts[j].r);
      sse += pts[j].w * err * err;
      sw += pts[j].w;
    }
    const meanSse = sse / sw;
    if (meanSse < bestSse) {
      bestSse = meanSse;
      best = { r, a: fit.a, b: fit.b, sse: meanSse };
    }
  }
  if (best === null) return null;

  // Prefer the least saturating ceiling that fits nearly as well as the
  // best: growth is only capped when the data itself shows the bend.
  let chosen = best;
  for (const r of ceilings) {
    if (r <= best.r) continue;
    const y = pts.map((p) => {
      const f = p.r / r;
      return Math.log(f / (1 - f));
    });
    const fit = wls(pts, y);
    if (fit === null || fit.b < 0) continue;
    let sse = 0;
    let sw = 0;
    for (let j = 0; j < pts.length; j++) {
      const pred = r * sigmoid(fit.a + fit.b * pts[j].u);
      const err = logit(pred) - logit(pts[j].r);
      sse += pts[j].w * err * err;
      sw += pts[j].w;
    }
    if (sse / sw <= 1.02 * bestSse) chosen = { r, a: fit.a, b: fit.b, sse: sse / sw };
  }

  const median = clampPct(chosen.r * sigmoid(chosen.a + chosen.b * uTarget));
  const sRes = Math.sqrt(Math.max(0, chosen.sse));
  const dU = uTarget - uLast;
  const sigma = Math.min(
    SIGMA_MAX,
    Math.max(
      SIGMA_MIN,
      Math.sqrt(sRes * sRes + (EXTRAP_SIGMA_PER_U * dU) ** 2 + REFRESH_SIGMA ** 2),
    ),
  );
  return { median, sigma };
}

/**
 * The distinct reads for one trophy: the first read, every read where the
 * rate changed from the previous kept read, and the last read. Long
 * stretches where Exophase showed the same page collapse to one point, so
 * the fit is not swamped by the refresh clock. Weights halve every 24 h
 * back from the last read.
 */
function trophyPoints(c: Case, i: number): { pts: Pt[]; lastRate: number } {
  const h = c.history;
  const last = h[h.length - 1];
  const lastRate = last.rates[i] ?? 0;
  const tLast = Date.parse(last.at);
  const pts: Pt[] = [];
  let prevRate: number | null = null;
  for (let j = 0; j < h.length; j++) {
    const r = h[j].rates[i];
    if (r === null || r === undefined) continue;
    const isLast = j === h.length - 1;
    const changed = prevRate === null || r !== prevRate;
    if (changed || isLast) {
      const ageH = Math.max(0, (tLast - Date.parse(h[j].at)) / 3_600_000);
      pts.push({ u: uOf(h[j].at), r, w: 2 ** (-ageH / WEIGHT_HALF_LIFE_H) });
      prevRate = r;
    }
  }
  return { pts, lastRate };
}

/** One trophy's five quantiles at the target, from a median and logit spread. */
function quantiles(median: number, sigma: number): number[] {
  const l = logit(median);
  const out: number[] = [];
  for (const z of Z) {
    const q = round4(100 * sigmoid(l + z * sigma));
    out.push(out.length === 0 ? q : Math.max(q, out[out.length - 1]));
  }
  return out;
}

/** Forecast one trophy: the fitted curve, or the flat fallback. */
function forecastTrophy(c: Case, i: number, uTarget: number): number[] {
  const { pts, lastRate } = trophyPoints(c, i);
  const fit = fitCurve(pts, uTarget);
  if (fit !== null) return quantiles(fit.median, fit.sigma);
  // No movement to fit: stay where the last read put it, with the house
  // baseline's random-walk spread, floored so a never-moved trophy still
  // admits it will move by the target.
  const days = c.horizon_minutes / 1440;
  const sigma = Math.max(FLAT_SIGMA * Math.sqrt(days), FALLBACK_SIGMA_MIN);
  return quantiles(clampPct(lastRate), sigma);
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const uTarget = uOf(c.target);
  const trophies = c.trophies.map((t, i) => ({
    id: t.id,
    q: forecastTrophy(c, i, uTarget),
  }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
