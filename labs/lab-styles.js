/* lab-styles.js — browser port of the LAB cell-type poster
 * (portrait-render/type_poster.py, which draws with render.py's type-level styles).
 *
 * parallel-styles.js ports the single-neuron sheet (neuron_sheet.py). The LAB
 * poster is the original multi-neuron render() in portrait-render/render.py:
 * every neuron of one type overlaid, with the original five styles (spectral
 * gives each neuron its own hue, constellation draws root -> 24 farthest
 * points per neuron). This file ports that, reusing only the skeleton
 * pre-processing from parallel-styles.js (prepare = swc_chains + long-edge cut
 * + _smooth, verified equal to load_swc), whose behaviour is not changed.
 *
 * Every colour, alpha, width and ordering below is copied from render.py; the
 * sheet layout (rect, text lines) and per-hemisphere outlier pruning from
 * type_poster.py. If either changes, change this file with it.
 *
 * API (window.LabStyles):
 *   STYLES, BG, FG
 *   parseSwc(text)                       -> skeleton object for PS.prepare
 *   loadSkeletons(ids, {concurrency, onProgress, isStale})
 *                                        -> [{id, sk}] in the order of ids, failures skipped
 *   neuronsFrom(list)                    -> [{id, ps, P, depth, rads, n}]
 *   dropOutliers(neurons)                -> render.drop_outliers (z=4.5, min_family=5)
 *   pruneBySide(neurons, sides)          -> type_poster.prune_per_side (sides: {id: "L"|"R"|"o"})
 *   orient([w,h], neurons)               -> render._orient: paper turned to match the data
 *   drawPoster(canvas, neurons, style, {paper:[w,h], label:{title, sub, caption, drawn, total}, ss})
 */
(function(){
"use strict";
const PS = window.ParallelStyles;
if(!PS){ return; }
const {percentile, norm, project} = PS._internal;
const autoTicks = PS._internal.autoTicks;
const STYLES = ["ink","spectral","blueprint","duotone","constellation"];
const BG = PS.BG, FG = PS.FG;           // identical to the (bg, fg) render.py's styles return

// type_poster.RECT, render._fit_limits(pad=0.04 default)
const LAYOUT = { rect:[0.07, 0.12, 0.86, 0.78], pad:0.04 };
const REF_DIAGONAL = Math.hypot(10.0, 13.0);
const MIN_LW_PT = 0.25;
const CREDIT = "MaleCNS v1.0  ·  data acquired and analyzed by the FlyEM Project Team " +
  "at HHMI-Janelia, the Cambridge Connectomics Group and Google Research" +
  "  ·  adapted  ·  CC BY 4.0  ·  creativecommons.org/licenses/by/4.0/";
const FONT = "'DejaVu Sans', Verdana, sans-serif";

// --- data ---------------------------------------------------------------------
const MEDIA = id => "https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o/" +
  encodeURIComponent("v1.0/segmentation/skeletons-malecns/skeletons-swc/" + id + ".swc") +
  "?alt=media";
function parseSwc(text){                 // same fields as index.html parse() needs for PS.prepare
  const ids=[],xs=[],ys=[],zs=[],rs=[],pa=[];
  for(const ln of text.split("\n")){
    if(!ln || ln[0]==="#") continue;
    const f=ln.trim().split(/\s+/); if(f.length<7) continue;
    ids.push(+f[0]); xs.push(+f[2]); ys.push(+f[3]); zs.push(+f[4]); rs.push(+f[5]); pa.push(+f[6]);
  }
  return {ids,xs,ys,zs,rs,pa};
}
const cache = new Map();
function fetchSkeleton(id){
  if(cache.has(id)) return cache.get(id);
  const p = fetch(MEDIA(id)).then(r => { if(!r.ok) throw new Error("HTTP "+r.status); return r.text(); })
    .then(parseSwc);
  p.catch(() => cache.delete(id));       // a failed fetch may be retried later
  cache.set(id, p); return p;
}
async function loadSkeletons(ids, o){
  o = o || {};
  const conc = o.concurrency || 4, out = new Array(ids.length);
  let next = 0, done = 0, failed = 0;
  async function worker(){
    while(next < ids.length){
      const k = next++;
      if(o.isStale && o.isStale()) return;
      try{ const sk = await fetchSkeleton(ids[k]); out[k] = {id:ids[k], sk}; }
      catch(e){ failed++; }
      done++;
      if(o.onProgress && !(o.isStale && o.isStale())) o.onProgress(done, ids.length, failed);
    }
  }
  const ws = []; for(let i=0;i<Math.min(conc, ids.length);i++) ws.push(worker());
  await Promise.all(ws);
  return out.filter(Boolean);
}
function neuronsFrom(list){
  const out = [];
  for(const {id, sk} of list){
    const ps = PS.prepare(sk);
    if(!ps.n) continue;
    const {P, depth} = project(ps);
    out.push({id, ps, P, depth, rads: ps.rads, n: ps.n});
  }
  return out;
}

// --- render.outlier_mask ------------------------------------------------------
function robustZ(v){
  const med = percentile(v, 50);
  const dev = v.map(x => Math.abs(x-med));
  const mad = percentile(dev, 50);
  if(mad <= 1e-9){ const den = Math.max(Math.abs(med), 1.0)*0.5; return v.map(x => Math.abs(x-med)/den); }
  return v.map(x => Math.abs(x-med)/(1.4826*mad));
}
function outlierMask(neurons, z, minFamily){
  z = z || 4.5; minFamily = minFamily || 5;
  const n = neurons.length, mask = new Array(n).fill(false);
  if(n < minFamily) return mask;
  const cx=[],cy=[],cz=[],ext=[],nseg=[];
  for(const nr of neurons){
    const S = nr.ps.segs, m = nr.n*2;
    let sx=0,sy=0,sz=0, x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity,z0=Infinity,z1=-Infinity;
    for(let j=0;j<m;j++){ const x=S[j*3], y=S[j*3+1], w=S[j*3+2];
      sx+=x; sy+=y; sz+=w;
      if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; if(w<z0)z0=w; if(w>z1)z1=w; }
    cx.push(sx/m); cy.push(sy/m); cz.push(sz/m);
    ext.push(Math.hypot(x1-x0, y1-y0, z1-z0)); nseg.push(nr.n);
  }
  const mx = percentile(cx,50), my = percentile(cy,50), mz = percentile(cz,50);
  const d = cx.map((_,i) => Math.hypot(cx[i]-mx, cy[i]-my, cz[i]-mz));
  const zd = robustZ(d), ze = robustZ(ext), zn = robustZ(nseg);
  for(let i=0;i<n;i++) mask[i] = zd[i] > z || ze[i] > z || zn[i] > z;
  const cnt = mask.filter(Boolean).length;
  if(cnt > Math.floor(n*2/3)) return new Array(n).fill(false);
  return mask;
}
function dropOutliers(neurons){
  const m = outlierMask(neurons);
  return neurons.filter((_, i) => !m[i]);
}
// type_poster.prune_per_side: outlier_mask run once per hemisphere
// (somaSide L / R / other), order preserved
function pruneBySide(neurons, sides){
  const keep = new Array(neurons.length).fill(false);
  for(const s of ["L", "R", "o"]){
    const idx = [];
    neurons.forEach((nr, i) => { const v = sides[nr.id]; if((v === "L" || v === "R" ? v : "o") === s) idx.push(i); });
    if(!idx.length) continue;
    const m = outlierMask(idx.map(i => neurons[i]));
    idx.forEach((i, k) => { if(!m[k]) keep[i] = true; });
  }
  return neurons.filter((_, i) => keep[i]);
}

// --- framing ------------------------------------------------------------------
function extent(neurons){
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(const nr of neurons){ const P = nr.P;
    for(let i=0;i<P.length;i+=2){ const x=P[i], y=P[i+1];
      if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; } }
  return {x0,x1,y0,y1};
}
function orient(size, neurons){            // render._orient
  const e = extent(neurons);
  const dataLand = (e.x1-e.x0)/Math.max(e.y1-e.y0, 1e-9) > 1.0;
  const paperLand = size[0] > size[1];
  if(dataLand !== paperLand && size[0] !== size[1]) return [size[1], size[0]];
  return [size[0], size[1]];
}

function hexA(hex, a){
  const n = parseInt(hex.slice(1),16);
  return "rgba("+(n>>16&255)+","+(n>>8&255)+","+(n&255)+","+a+")";
}
function hsvToRgb(h, s, v){                  // colorsys.hsv_to_rgb
  if(s === 0) return [v,v,v];
  let i = Math.floor(h*6); const f = h*6 - i;
  const p = v*(1-s), q = v*(1-s*f), t = v*(1-s*(1-f));
  i = ((i % 6) + 6) % 6;
  return [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i];
}

function paint(cv, neurons, style, opts){
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  const paper = opts.paper;                              // inches, already oriented
  const SCALE = Math.hypot(paper[0], paper[1]) / REF_DIAGONAL;
  const ptPx = W / (paper[0]*72);
  const lwPx = v => Math.max(v*SCALE, MIN_LW_PT) * ptPx;
  const rect = LAYOUT.rect;

  ctx.save();
  ctx.setTransform(1,0,0,1,0,0); ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = BG[style]; ctx.fillRect(0,0,W,H);
  if(!neurons.length){ ctx.restore(); return; }

  // _fit_limits(ax, pts, size, rect, pad=0.04)
  const e = extent(neurons), pad = LAYOUT.pad;
  let dw = Math.max(e.x1-e.x0,1e-9)*(1+2*pad), dh = Math.max(e.y1-e.y0,1e-9)*(1+2*pad);
  const target = (paper[0]*rect[2]) / (paper[1]*rect[3]);
  if(dw/dh < target) dw = dh*target; else dh = dw/target;
  const ccx = (e.x0+e.x1)/2, ccy = (e.y0+e.y1)/2;
  const xl0 = ccx-dw/2, xl1 = ccx+dw/2, yl0 = ccy-dh/2, yl1 = ccy+dh/2;
  const AX = rect[0]*W, AW = rect[2]*W, AY = H - (rect[1]+rect[3])*H, AH = rect[3]*H;
  const sx = AW/(xl1-xl0), sy = AH/(yl1-yl0);
  const X = x => AX + (x-xl0)*sx;
  const Y = y => AY + (yl1-y)*sy;
  const seg = (P, i, dx, dy) => {
    ctx.beginPath();
    ctx.moveTo(X(P[i*4]+dx), Y(P[i*4+1]+dy));
    ctx.lineTo(X(P[i*4+2]+dx), Y(P[i*4+3]+dy));
    ctx.stroke();
  };

  let ticksX = null, ticksY = null;
  if(style === "blueprint"){
    const lab = 5*SCALE;                                   // _fs(5) pt
    const nbx = Math.max(Math.min(Math.floor(rect[2]*paper[0]*72/(lab*3)), 9), 1);
    const nby = Math.max(Math.min(Math.floor(rect[3]*paper[1]*72/(lab*2)), 9), 1);
    ticksX = autoTicks(xl0, xl1, nbx); ticksY = autoTicks(yl0, yl1, nby);
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(AX, AY, AW, AH); ctx.clip();
  ctx.fillStyle = BG[style]; ctx.fillRect(AX, AY, AW, AH);
  const N = neurons.length;

  if(style === "ink"){
    ctx.lineCap = "round"; ctx.strokeStyle = hexA("#1b1815", 0.85);
    for(const nr of neurons){
      const w = norm(nr.rads);
      for(let i=0;i<nr.n;i++){ ctx.lineWidth = lwPx(0.15 + 0.9*w[i]); seg(nr.P, i, 0, 0); }
    }
  }
  else if(style === "spectral"){
    ctx.lineCap = "round";
    const n = Math.max(N, 1);
    neurons.forEach((nr, k) => {
      const hue = (0.58 + 0.72*k/n) % 1.0;
      const d = norm(nr.depth), w = norm(nr.rads);
      for(let i=0;i<nr.n;i++){
        const dd = d[i], c = hsvToRgb(hue, 0.55 + 0.35*dd, 0.35 + 0.65*dd);
        ctx.strokeStyle = "rgba("+Math.round(c[0]*255)+","+Math.round(c[1]*255)+","+Math.round(c[2]*255)+","+(0.25+0.55*dd)+")";
        ctx.lineWidth = lwPx(0.2 + 1.1*w[i]);
        seg(nr.P, i, 0, 0);
      }
    });
  }
  else if(style === "blueprint"){
    ctx.lineCap = "butt";
    ctx.strokeStyle = hexA("#1e4a72", 0.6); ctx.lineWidth = lwPx(0.4);      // grid, axis below
    for(const t of ticksX){ const x=X(t); ctx.beginPath(); ctx.moveTo(x,AY); ctx.lineTo(x,AY+AH); ctx.stroke(); }
    for(const t of ticksY){ const y=Y(t); ctx.beginPath(); ctx.moveTo(AX,y); ctx.lineTo(AX+AW,y); ctx.stroke(); }
    for(const nr of neurons){                               // per neuron: underlay, then hairline
      ctx.strokeStyle = hexA("#2ec5ff", 0.10); ctx.lineWidth = lwPx(1.6);
      for(let i=0;i<nr.n;i++) seg(nr.P, i, 0, 0);
      ctx.strokeStyle = hexA("#7fe3ff", 0.9); ctx.lineWidth = lwPx(0.35);
      for(let i=0;i<nr.n;i++) seg(nr.P, i, 0, 0);
    }
  }
  else if(style === "duotone"){
    // offset = 0.6% of the *first* neuron's projected span, as render.py does
    const P0 = neurons[0].P;
    let px0=Infinity,px1=-Infinity,py0=Infinity,py1=-Infinity;
    for(let i=0;i<P0.length;i+=2){ const x=P0[i],y=P0[i+1];
      if(x<px0)px0=x; if(x>px1)px1=x; if(y<py0)py0=y; if(y>py1)py1=y; }
    const ox = (px1-px0)*0.006, oy = (py1-py0)*0.006;
    ctx.lineCap = "butt"; ctx.lineWidth = lwPx(0.9);
    for(const nr of neurons){
      ctx.strokeStyle = hexA("#ff4f37", 0.55); for(let i=0;i<nr.n;i++) seg(nr.P, i, ox, oy);
      ctx.strokeStyle = hexA("#1c3f94", 0.55); for(let i=0;i<nr.n;i++) seg(nr.P, i, -ox, -oy);
    }
  }
  else if(style === "constellation"){
    // root = pts[argmax(rad)] where pts is the (2N,2) endpoint array and rad has
    // N entries (render.py indexes the flattened points with a segment index;
    // ported as is). far = the 24 points farthest from the root.
    const lines = [], tips = [], roots = [];
    for(const nr of neurons){
      const P = nr.P, R = nr.rads, m = nr.n*2;
      let k = 0; for(let i=1;i<nr.n;i++) if(R[i] > R[k]) k = i;
      const rx = P[k*2], ry = P[k*2+1];
      roots.push([rx, ry]);
      const d = new Float64Array(m), idx = new Array(m);
      for(let j=0;j<m;j++){ d[j] = Math.hypot(P[j*2]-rx, P[j*2+1]-ry); idx[j] = j; }
      idx.sort((a,b) => d[a]-d[b]);
      for(const j of idx.slice(Math.max(0, m-24))){ tips.push([P[j*2], P[j*2+1]]); lines.push([rx, ry, P[j*2], P[j*2+1]]); }
    }
    const dot = (x, y, r) => { ctx.beginPath(); ctx.arc(X(x), Y(y), r, 0, Math.PI*2); ctx.fill(); };
    // matplotlib draws by zorder: scatter collections (1) before plot lines (2); roots have zorder 5
    ctx.fillStyle = hexA("#f3d79a", 0.55);
    const rTip = Math.sqrt(0.6)*SCALE/2*ptPx;                 // s = 0.6*SCALE^2 pt^2 (area) -> radius
    for(const t of tips) dot(t[0], t[1], rTip);
    ctx.strokeStyle = hexA("#d9a441", 0.30); ctx.lineWidth = lwPx(0.22); ctx.lineCap = "round";
    for(const l of lines){ ctx.beginPath(); ctx.moveTo(X(l[0]), Y(l[1])); ctx.lineTo(X(l[2]), Y(l[3])); ctx.stroke(); }
    ctx.fillStyle = hexA("#ffe9b0", 0.95);
    const rRoot = Math.sqrt(14)*SCALE/2*ptPx;
    for(const r of roots) dot(r[0], r[1], rRoot);
  }
  ctx.restore();

  if(style === "blueprint"){
    ctx.strokeStyle = "#1e4a72"; ctx.lineWidth = 0.8*ptPx; ctx.lineCap = "butt";
    ctx.strokeRect(AX, AY, AW, AH);
    ctx.strokeStyle = "#4e88b8"; ctx.lineWidth = lwPx(0.5);
    const tl = 3*SCALE*ptPx, padPx = 3.5*ptPx;
    ctx.fillStyle = "#4e88b8";
    ctx.font = (5*SCALE*ptPx).toFixed(2)+"px "+FONT;
    const lbl = v => { const r = Math.round(v*1e6)/1e6; return (r<0?"−":"") + String(Math.abs(r)); };
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for(const t of ticksX){ const x=X(t); ctx.beginPath(); ctx.moveTo(x,AY+AH); ctx.lineTo(x,AY+AH+tl); ctx.stroke();
      ctx.fillText(lbl(t), x, AY+AH+tl+padPx); }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for(const t of ticksY){ const y=Y(t); ctx.beginPath(); ctx.moveTo(AX,y); ctx.lineTo(AX-tl,y); ctx.stroke();
      ctx.fillText(lbl(t), AX-tl-padPx, y); }
  }

  if(opts.label){
    // type_poster.render_type fig.text calls: (x, baseline y, size pt, alpha, align)
    const fg = FG[style], L = opts.label;
    const txt = (s, x, y, pt, a, align, weight) => {
      if(!s) return;
      ctx.globalAlpha = a; ctx.textAlign = align;
      ctx.font = (weight||"400")+" "+(pt*SCALE*ptPx).toFixed(2)+"px "+FONT;
      ctx.fillText(s, x*W, H*(1-y));
    };
    ctx.fillStyle = fg; ctx.textBaseline = "alphabetic";
    txt(String(L.title||""), 0.07, 0.950, 27, 1, "left", "300");
    txt(String(L.sub||"").toUpperCase(), 0.07, 0.922, 7.5, 0.75, "left");
    txt(String(L.caption||""), 0.07, 0.074, 7, 0.85, "left");
    const drawn = L.drawn === L.total ? L.drawn + " neurons" : L.drawn + " of " + L.total + " neurons drawn";
    txt(drawn + "  ·  MaleCNS v1.0  ·  frontal view", 0.07, 0.052, 5.5, 0.6, "left");
    txt(CREDIT, 0.93, 0.030, 4.3, 0.55, "right");
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

// Draw at >= opts.ss px on the long side off screen, then halve down step by
// step (same reason as parallel-styles.js drawSheet: sub-pixel strokes drawn
// directly on a small canvas come out visibly heavier than a downsized print).
let _off = null;
function drawPoster(cv, neurons, style, opts){
  opts = opts || {};
  if(STYLES.indexOf(style) < 0) style = "ink";
  const ss = opts.ss || 2400, W = cv.width, H = cv.height, L = Math.max(W, H);
  if(L >= ss || typeof document === "undefined"){ paint(cv, neurons, style, opts); return; }
  let k = 1; while(L*k < ss) k *= 2;
  if(!_off) _off = document.createElement("canvas");
  _off.width = W*k; _off.height = H*k;
  paint(_off, neurons, style, opts);
  let src = _off, w = W*k, h = H*k;
  while(w > W){
    const nw = w/2, nh = h/2;
    const t = (nw === W) ? cv : document.createElement("canvas");
    if(t !== cv){ t.width = nw; t.height = nh; }
    const c = t.getContext("2d");
    c.imageSmoothingEnabled = true; c.imageSmoothingQuality = "high";
    c.setTransform(1,0,0,1,0,0); c.globalAlpha = 1; c.globalCompositeOperation = "copy";
    c.drawImage(src, 0, 0, w, h, 0, 0, nw, nh);
    c.globalCompositeOperation = "source-over";
    src = t; w = nw; h = nh;
  }
  _off.width = 1; _off.height = 1;             // release the big buffer
}

window.LabStyles = {STYLES, BG, FG, LAYOUT, CREDIT, parseSwc, loadSkeletons, neuronsFrom,
  outlierMask, dropOutliers, pruneBySide, orient, drawPoster};
})();
