/**
 * HRP Benchmark — Bun
 *
 * Faithful port of the reference C implementation. The synthetic price data is
 * generated with the same 64-bit LCG (via BigInt) so prices — and therefore the
 * resulting HRP weights — are bit-identical across every language in the suite.
 * Generation is NOT timed; only the five HRP stages are.
 *
 * The average-linkage stage uses the O(n²) nearest-neighbor chain algorithm
 * (Müllner 2011), ported directly from the C reference. No external dependencies
 * are required; the linkage is computed natively to produce bit-identical results
 * with the C/Python/Node variants of this benchmark.
 */

// ── Synthetic prices: 64-bit LCG (BigInt), identical to the C reference ──

function generatePrices(n, days) {
  const MASK = (1n << 64n) - 1n;
  const MULT = 6364136223846793005n;
  const MAXF = Number(MASK);
  let seed = 42n;
  const prices = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * MULT + 1n) & MASK;
    const start = 100.0 + Number(seed % 900n);
    const vol = 0.01 + Number(seed % 50n) * 0.001;
    const row = new Float64Array(days);
    row[0] = start;
    for (let t = 1; t < days; t++) {
      seed = (seed * MULT + 1n) & MASK;
      const u = Number(seed) / MAXF - 0.5;
      const ret = vol * u * 0.816;
      row[t] = row[t - 1] * Math.exp(ret);
    }
    prices.push(row);
  }
  return prices;
}

// ── Stage 1: log returns ──

function logReturns(prices) {
  return prices.map((p) => {
    const r = new Float64Array(p.length - 1);
    for (let t = 0; t < p.length - 1; t++) r[t] = Math.log(p[t + 1] / p[t]);
    return r;
  });
}

// ── Stage 2: sample covariance (divide by T-1) ──

function covarianceMatrix(rets) {
  const n = rets.length, T = rets[0].length;
  const means = rets.map((r) => { let s = 0; for (let t = 0; t < T; t++) s += r[t]; return s / T; });
  const cov = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += (rets[i][t] - means[i]) * (rets[j][t] - means[j]);
      cov[i][j] = cov[j][i] = s / (T - 1);
    }
  return cov;
}

// ── Correlation + distance (computed, NOT timed — matches C) ──

function correlationMatrix(cov) {
  const n = cov.length;
  const stds = cov.map((_, i) => Math.sqrt(cov[i][i]));
  const corr = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      corr[i][j] = (stds[i] > 0 && stds[j] > 0) ? cov[i][j] / (stds[i] * stds[j]) : 0;
  return corr;
}

function distanceMatrix(corr) {
  const n = corr.length;
  const d = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const v = (1 - corr[i][j]) / 2;
      d[i][j] = Math.sqrt(v > 0 ? v : 0);
    }
  return d;
}

// ── Stage 3: average linkage (O(n²) NN-chain) + leaf-order extraction ──
//
// Direct port of the C reference implementation (Müllner 2011 NN-chain with
// Lance-Williams average update, followed by SciPy-style distance sort and
// union-find relabeling). Produces a linkage matrix Z identical to the C build,
// so the resulting leaf order — and therefore HRP weights — are bit-identical.

function averageLinkage(dist) {
  const n = dist.length;
  const D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) D[i * n + j] = dist[i][j];

  const size   = new Int32Array(n).fill(1);
  const active = new Uint8Array(n).fill(1);
  const chain  = new Int32Array(n);
  let clen = 0;

  const rawI = new Int32Array(n - 1);
  const rawJ = new Int32Array(n - 1);
  const rawD = new Float64Array(n - 1);
  const rawS = new Int32Array(n - 1);
  let m = 0;

  for (let s = 0; s < n - 1; s++) {
    if (clen === 0) {
      let start = -1;
      for (let i = 0; i < n; i++) if (active[i]) { start = i; break; }
      chain[clen++] = start;
    }
    let a, b = -1, mind;
    for (;;) {
      a = chain[clen - 1];
      b = -1; mind = 1e18;
      if (clen >= 2) { b = chain[clen - 2]; mind = D[a * n + b]; }
      for (let x = 0; x < n; x++) {
        if (!active[x] || x === a) continue;
        const d = D[a * n + x];
        if (d < mind) { mind = d; b = x; }
      }
      if (clen >= 2 && b === chain[clen - 2]) break;
      chain[clen++] = b;
    }
    clen -= 2;
    const x = a < b ? a : b, y = a < b ? b : a;
    const ns = size[x] + size[y];
    rawI[m] = x; rawJ[m] = y; rawD[m] = mind; rawS[m] = ns; m++;
    for (let k = 0; k < n; k++) {
      if (!active[k] || k === x || k === y) continue;
      const nd = (size[x] * D[x * n + k] + size[y] * D[y * n + k]) / ns;
      D[y * n + k] = nd; D[k * n + y] = nd;
    }
    size[y] = ns; active[x] = 0;
  }

  const idx = Array.from({ length: n - 1 }, (_, i) => i);
  idx.sort((a, b) => rawD[a] - rawD[b]);

  const parent = new Int32Array(2 * n);
  const usize  = new Int32Array(2 * n).fill(1);
  for (let i = 0; i < 2 * n; i++) parent[i] = i;

  function ufFind(x) {
    let root = x;
    while (parent[root] !== root) root = parent[root];
    while (parent[x] !== root) { const nx = parent[x]; parent[x] = root; x = nx; }
    return root;
  }

  const Zi = new Int32Array(n - 1);
  const Zj = new Int32Array(n - 1);
  const Zd = new Float64Array(n - 1);
  const Zs = new Int32Array(n - 1);
  let next = n;

  for (let k = 0; k < n - 1; k++) {
    const ri = idx[k];
    const xr = ufFind(rawI[ri]), yr = ufFind(rawJ[ri]);
    const lo = xr < yr ? xr : yr, hi = xr < yr ? yr : xr;
    Zi[k] = lo; Zj[k] = hi; Zd[k] = rawD[ri]; Zs[k] = usize[xr] + usize[yr];
    parent[xr] = next; parent[yr] = next;
    usize[next] = usize[xr] + usize[yr];
    next++;
  }
  return { Zi, Zj, Zd, Zs };
}

function linkageLeafOrder(dist) {
  const n = dist.length;
  const { Zi, Zj } = averageLinkage(dist);

  const order = [];
  const stack = [n + (n - 2)];
  while (stack.length) {
    const node = stack.pop();
    if (node < n) { order.push(node); continue; }
    const k = node - n;
    stack.push(Zj[k]);
    stack.push(Zi[k]);
  }
  return order;
}

// ── Stage 5: HRP recursive bisection weights ──

function clusterVar(cov, idx) {
  let v = 0;
  for (const a of idx) for (const b of idx) v += cov[a][b];
  return v / (idx.length * idx.length);
}

function hrpWeights(covQ, n) {
  const w = new Float64Array(n).fill(1);
  const queue = [Array.from({ length: n }, (_, i) => i)];
  while (queue.length) {
    const seg = queue.shift();
    if (seg.length <= 1) continue;
    const mid = Math.floor(seg.length / 2);
    const left = seg.slice(0, mid);
    const right = seg.slice(mid);
    const vL = clusterVar(covQ, left);
    const vR = clusterVar(covQ, right);
    const alpha = (1 / vL) / (1 / vL + 1 / vR);
    for (const i of left) w[i] *= alpha;
    for (const i of right) w[i] *= 1 - alpha;
    queue.push(left);
    queue.push(right);
  }
  let sum = 0;
  for (let i = 0; i < n; i++) sum += w[i];
  for (let i = 0; i < n; i++) w[i] /= sum;
  return w;
}

// ── Per-size benchmark ──

function fmt(us) {
  if (us < 1000) return us.toFixed(0).padStart(6) + "µs";
  if (us < 1e6) return (us / 1e3).toFixed(1).padStart(6) + "ms";
  return (us / 1e6).toFixed(2).padStart(6) + "s ";
}

function bench(n, days) {
  const prices = generatePrices(n, days);

  let t = performance.now();
  const rets = logReturns(prices);
  const tRet = (performance.now() - t) * 1000;

  t = performance.now();
  const cov = covarianceMatrix(rets);
  const tCov = (performance.now() - t) * 1000;

  const corr = correlationMatrix(cov);
  const dist = distanceMatrix(corr);

  t = performance.now();
  const order = linkageLeafOrder(dist);
  const tLink = (performance.now() - t) * 1000;

  t = performance.now();
  const covQ = order.map((oi) => order.map((oj) => cov[oi][oj]));
  const tQD = (performance.now() - t) * 1000;

  t = performance.now();
  const w = hrpWeights(covQ, n);
  const tW = (performance.now() - t) * 1000;

  const total = tRet + tCov + tLink + tQD + tW;
  console.log(
    "  " + String(n).padStart(6) + " │ " + fmt(tRet) + " │ " + fmt(tCov) + " │ " +
    fmt(tLink) + " │ " + fmt(tQD) + " │ " + fmt(tW) + " │ " + fmt(total)
  );
  return w;
}

// ── Main ──

console.log("╔═══════════════════════════════════════════════════════════════════╗");
console.log("║          HRP Benchmark — Bun                                      ║");
console.log("╚═══════════════════════════════════════════════════════════════════╝");
console.log("  365 daily observations per asset\n");
console.log(
  "  " + "N".padStart(6) + " │ " + "LogRet".padStart(8) + " │ " + "Cov".padStart(8) +
  " │ " + "Linkage".padStart(8) + " │ " + "QuasiD".padStart(8) + " │ " +
  "Weights".padStart(8) + " │ " + "TOTAL".padStart(8)
);
console.log("  " + "─".repeat(67));

const sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
const DAYS = 365;
let w10 = null;
for (const n of sizes) {
  const memGB = (n * n * 8) / 1024 ** 3;
  if (memGB > 4) {
    console.log(`  ${String(n).padStart(6)} │ skipped — would need ${memGB.toFixed(1)}GB for ${n}×${n} matrix`);
    continue;
  }
  const w = bench(n, DAYS);
  if (n === 10) w10 = w;
}

if (w10) {
  let sum = 0;
  for (const x of w10) sum += x;
  console.error(`verify N=10: w0=${w10[0].toFixed(12)} w1=${w10[1].toFixed(12)} w2=${w10[2].toFixed(12)} sum=${sum.toFixed(6)}`);
}
