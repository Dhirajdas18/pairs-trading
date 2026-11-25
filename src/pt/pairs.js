// Pair maths in the browser. Mirrors src/pt/pairs.py; tests compare the two through Node.
// Inputs are aligned arrays of numbers without gaps.

const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;

// MacKinnon (1994) response surface, constant case, keyed by number of I(1) variables (as pairs.py).
const MACKINNON_C = {
  1: { max: 2.74, min: -18.83, star: -1.61, small: [2.1659, 1.4412, 0.038269], large: [1.7339, 0.93202, -0.12745, -0.010368] },
  2: { max: 0.92, min: -18.86, star: -2.62, small: [2.92, 1.5012, 0.039796], large: [2.1945, 0.64695, -0.29198, -0.042377] },
};

// Standard normal cdf via erfc (W. J. Cody's rational approximations, double precision).
function erfc(x) {
  const ax = Math.abs(x);
  let r;
  if (ax < 0.5) {
    const t = x * x;
    const top = (((0.185777706184603153 * t + 3.16112374387056560) * t + 113.864154151050156) * t + 377.485237685302021) * t + 3209.37758913846947;
    const bot = (((t + 23.6012909523441209) * t + 244.024637934444173) * t + 1282.61652607737228) * t + 2844.23683343917062;
    return 1 - x * top / bot;
  }
  if (ax < 4) {
    const c = [5.64188496988670089e-1, 8.88314979438837594, 66.1191906371416295, 298.635138197400131, 881.952221241769090, 1712.04761263407058, 2051.07837782607147, 1230.33935479799725, 2.15311535474403846e-8];
    const d = [15.7449261107098347, 117.693950891312499, 537.181101862009858, 1621.38957456669019, 3290.79923573345963, 4362.61909014324716, 3439.36767414372164, 1230.33935480374942];
    let top = c[8] * ax, bot = ax;
    for (let i = 0; i < 7; i++) { top = (top + c[i]) * ax; bot = (bot + d[i]) * ax; }
    r = (top + c[7]) / (bot + d[7]);
  } else {
    const p = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2];
    const q = [2.56852019228982242, 1.87295284992346725, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3];
    const z = 1 / (ax * ax);
    let top = p[5] * z, bot = z;
    for (let i = 0; i < 4; i++) { top = (top + p[i]) * z; bot = (bot + q[i]) * z; }
    r = z * (top + p[4]) / (bot + q[4]);
    r = (0.564189583547756287 - r) / ax;
  }
  const xs = Math.trunc(ax * 16) / 16, del = (ax - xs) * (ax + xs);
  r *= Math.exp(-xs * xs) * Math.exp(-del);
  return x < 0 ? 2 - r : r;
}
const normCdf = (z) => 0.5 * erfc(-z / Math.SQRT2);

// OLS of y on the columns of X: coefficients and plain standard errors (residual variance on n - k).
function olsMulti(y, cols) {
  const n = y.length, k = cols.length;
  const A = cols.map((ci) => cols.map((cj) => { let s = 0; for (let t = 0; t < n; t++) s += ci[t] * cj[t]; return s; }));
  const inv = A.map((row, i) => row.map((_, j) => (i === j ? 1 : 0)));
  const M = A.map((row) => row.slice());
  for (let c = 0; c < k; c++) {  // Gauss-Jordan with partial pivoting
    let p = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]]; [inv[c], inv[p]] = [inv[p], inv[c]];
    const d = M[c][c];
    for (let j = 0; j < k; j++) { M[c][j] /= d; inv[c][j] /= d; }
    for (let r = 0; r < k; r++) {
      if (r === c) continue;
      const f = M[r][c];
      for (let j = 0; j < k; j++) { M[r][j] -= f * M[c][j]; inv[r][j] -= f * inv[c][j]; }
    }
  }
  const xty = cols.map((ci) => { let s = 0; for (let t = 0; t < n; t++) s += ci[t] * y[t]; return s; });
  const coef = inv.map((row) => row.reduce((s, v, j) => s + v * xty[j], 0));
  let ssr = 0;
  for (let t = 0; t < n; t++) { let f = 0; for (let j = 0; j < k; j++) f += coef[j] * cols[j][t]; ssr += (y[t] - f) ** 2; }
  const s2 = ssr / (n - k);
  return { coef, se: inv.map((row, i) => Math.sqrt(s2 * row[i])) };
}

// [alpha, beta] from OLS of log A on log B with an intercept.
function hedgeRatio(a, b) {
  const ma = mean(a), mb = mean(b);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < a.length; i++) { sxy += (b[i] - mb) * (a[i] - ma); sxx += (b[i] - mb) ** 2; }
  const beta = sxy / sxx;
  return [ma - beta * mb, beta];
}

const spread = (a, b, alpha, beta) => a.map((v, i) => v - alpha - beta * b[i]);

// Full-sample z-score (window null) or rolling over the last `window` values up to t (null before).
function zscore(s, window = null) {
  const sd = (w) => { const m = mean(w); return [m, Math.sqrt(w.reduce((a, v) => a + (v - m) ** 2, 0) / (w.length - 1))]; };
  if (window === null) { const [m, d] = sd(s); return s.map((v) => (v - m) / d); }
  return s.map((v, t) => { if (t < window - 1) return null; const [m, d] = sd(s.slice(t - window + 1, t + 1)); return (v - m) / d; });
}

// Days for a deviation to halve, from ds_t = a + b s_{t-1}; Infinity when b >= 0.
function halfLife(s) {
  const lag = s.slice(0, -1), ds = s.slice(1).map((v, i) => v - s[i]);
  const ml = mean(lag), md = mean(ds);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < lag.length; i++) { sxy += (lag[i] - ml) * (ds[i] - md); sxx += (lag[i] - ml) ** 2; }
  const b = sxy / sxx;
  if (b >= 0) return Infinity;
  if (b <= -1) return 0;
  return -Math.LN2 / Math.log(1 + b);
}

function mackinnonP(t, nVars = 2) {
  const c = MACKINNON_C[nVars];
  if (t > c.max) return 1;
  if (t < c.min) return 0;
  const coef = t <= c.star ? c.small : c.large;
  return normCdf(coef.reduce((s, k, i) => s + k * t ** i, 0));
}

// ADF t-statistic on x_{t-1} with `lags` lagged differences, and MacKinnon's p-value.
function adf(x, lags = 1, constant = true, nVars = 1) {
  const dx = x.slice(1).map((v, i) => v - x[i]);
  const y = dx.slice(lags), cols = [x.slice(lags, -1)];
  for (let j = 1; j <= lags; j++) cols.push(dx.slice(lags - j, dx.length - j));
  if (constant) cols.push(new Array(y.length).fill(1));
  const { coef, se } = olsMulti(y, cols);
  const tstat = coef[0] / se[0];
  return { tstat, pvalue: mackinnonP(tstat, nVars) };
}

// Engle-Granger two-step test of log A on log B (as statsmodels coint, trend "c", fixed lags).
function engleGranger(a, b, lags = 1) {
  const [alpha, beta] = hedgeRatio(a, b), s = spread(a, b, alpha, beta);
  const { tstat, pvalue } = adf(s, lags, false, 2);
  return { alpha, beta, tstat, pvalue, half_life: halfLife(s) };
}

// Spread position at each close (-1, 0, +1) from the z-score at that close; null z means flat.
function positions(z, entry = 2, exit = 0.5, stop = null) {
  let p = 0, stopped = false;
  return z.map((v) => {
    if (v === null || Number.isNaN(v)) { p = 0; stopped = false; }
    else if (stop !== null && Math.abs(v) > stop) { p = 0; stopped = true; }
    else if (stopped) { if (Math.abs(v) < exit) stopped = false; }
    else if (v > entry) p = -1;
    else if (v < -entry) p = 1;
    else if (Math.abs(v) < exit) p = 0;
    return p;
  });
}

const COST_PER_SIDE = 0.0005;

// Daily returns of the z-score rule after costs; position at close t earns returns t to t+1.
function backtest(a, b, beta, z, entry = 2, exit = 0.5, stop = null, costPerSide = COST_PER_SIDE) {
  const pos = positions(z, entry, exit, stop), g = 1 + Math.abs(beta);
  const wa = pos.map((p) => p / g), wb = pos.map((p) => -p * beta / g);
  const n = a.length, r = new Array(n).fill(0);
  let trades = 0;
  for (let t = 0; t < n; t++) {
    const prev = t ? pos[t - 1] : 0;
    if (pos[t] !== 0 && pos[t] !== prev) trades++;
    if (t === 0) continue;
    const ra = Math.expm1(a[t] - a[t - 1]), rb = Math.expm1(b[t] - b[t - 1]);
    const turn = Math.abs(wa[t - 1] - (t > 1 ? wa[t - 2] : 0)) + Math.abs(wb[t - 1] - (t > 1 ? wb[t - 2] : 0));
    r[t] = wa[t - 1] * ra + wb[t - 1] * rb - costPerSide * turn;
  }
  let e = 1;
  const equity = r.map((v) => (e *= 1 + v));
  return { returns: r, positions: pos, trades, equity };
}

// Pearson correlation.
function corr(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
}

if (typeof module !== "undefined") {
  module.exports = { hedgeRatio, spread, zscore, halfLife, mackinnonP, adf, engleGranger, positions, backtest, corr, normCdf };
}
