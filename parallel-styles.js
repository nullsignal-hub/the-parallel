/* parallel-styles.js — browser port of the five print styles.
 *
 * This is a line-for-line port of portrait-render/render.py (and, for the
 * single-neuron spectral + constellation looks, of neuron_sheet.py's
 * style_spectral_single / style_constellation_single):
 *   swc_chains / _smooth / load_swc  (skeleton -> smoothed 2-point segments)
 *   project("frontal"), _norm, _lw, SCALE
 *   style_ink / style_spectral / style_blueprint / style_duotone /
 *   style_constellation
 * The print is rendered server-side by that Python code. The preview here must
 * not disagree with it, so nothing below is a "look" decision: every colour,
 * alpha, width and ordering is copied from render.py. If render.py changes,
 * change this file with it.
 *
 * Input: the page's parsed SWC object, which must carry
 *   {ids, xs, ys, zs, rs, pa}  (node id, x, y, z, radius, parent id, file order).
 *
 * API (window.ParallelStyles):
 *   STYLES                       ["ink","spectral","blueprint","duotone","constellation"]
 *   BG[style]                    paper colour of that style
 *   prepare(sk)                  -> {segs:Float64Array(N*6), rads:Float64Array(N)} (cached on sk)
 *   drawSheet(canvas, sk, style, opts)
 *       Draws the whole 12x12in sheet onto a square canvas, laid out like
 *       portrait-render/neuron_sheet.py (the script that renders the print).
 *       opts.label = {type, bodyId, subtitle}   sheet text (omit for art only)
 *                    subtitle: use subtitleFor(rosterEntry)
 *       opts.paper = 12               sheet size in inches (square)
 *       opts.direct = true            skip the draw-large-then-downscale step
 *   subtitleFor(n)               neuron_sheet.subtitle_for (kind + hemisphere)
 */
(function(){
"use strict";

const STYLES = ["ink","spectral","blueprint","duotone","constellation"];

// --- sheet layout -----------------------------------------------------------
// Figure-fraction rectangle of the drawing area (matplotlib add_axes rect:
// left, bottom, width, height) and how the neuron is framed inside it.
// Kept in one place so it can be matched to the server's sheet layout.
// Matches portrait-render/neuron_sheet.py (the script that renders the print):
//   RECT = [0.08, 0.13, 0.84, 0.74], R._fit_limits(ax, pts, size, RECT, pad=0.06)
const LAYOUT = {
  rect: [0.08, 0.13, 0.84, 0.74],
  frame: "fit",         // "fit"  = render.py _fit_limits(pad) on every drawn point
                        // "hero" = portrait.py (square box, half = max span * 0.72)
  half: 0.72,
  pad: 0.06,
};
// render.CREDIT, verbatim
const CREDIT = "MaleCNS v1.0  ·  data acquired and analyzed by the FlyEM Project Team " +
  "at HHMI-Janelia, the Cambridge Connectomics Group and Google Research" +
  "  ·  adapted  ·  CC BY 4.0  ·  creativecommons.org/licenses/by/4.0/";
// neuron_sheet.subtitle_for: roster kind + hemisphere word
const SIDE_WORD = {L:"left hemisphere", R:"right hemisphere", M:"midline"};
function subtitleFor(n){
  if(!n) return "";
  const parts = [String(n.kind || "").trim()];
  if(SIDE_WORD[n.side]) parts.push(SIDE_WORD[n.side]);
  return parts.filter(Boolean).join("  ·  ");
}

// render.py constants
const REF_DIAGONAL = Math.hypot(10.0, 13.0);
const MIN_LW_PT = 0.25;

// --- numpy helpers ------------------------------------------------------------
function percentile(arr, q){                 // numpy default ("linear")
  const n = arr.length; if(!n) return 0;
  const a = Float64Array.from(arr).sort();
  const pos = (q/100)*(n-1), lo = Math.floor(pos), hi = Math.min(lo+1, n-1);
  return a[lo] + (a[hi]-a[lo])*(pos-lo);
}
function median(arr){ return percentile(arr, 50); }
function norm(v){                            // render._norm
  const lo = percentile(v, 2), hi = percentile(v, 98);
  const d = Math.max(hi-lo, 1e-9), out = new Float64Array(v.length);
  for(let i=0;i<v.length;i++){ const t=(v[i]-lo)/d; out[i] = t<0?0:t>1?1:t; }
  return out;
}
function hsvToRgb(h, s, v){                  // colorsys.hsv_to_rgb
  if(s === 0) return [v,v,v];
  let i = Math.floor(h*6); const f = h*6 - i;
  const p = v*(1-s), q = v*(1-s*f), t = v*(1-s*(1-f));
  i = ((i % 6) + 6) % 6;
  return [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i];
}

// --- skeleton -> segments  (render.swc_chains + load_swc) ---------------------
function swcChains(sk){
  const nid = sk.ids, par = sk.pa, N = nid.length;
  const index = new Map();
  for(let i=0;i<N;i++) index.set(nid[i], i);           // later rows overwrite, as a dict does
  const children = new Map(); const roots = [];        // Map keeps insertion order, as a dict does
  for(let i=0;i<N;i++){
    const j = index.has(par[i]) ? index.get(par[i]) : -1;
    if(j < 0 || j === i) roots.push(i);
    else { let c = children.get(j); if(!c){ c=[]; children.set(j,c); } c.push(i); }
  }
  const starts = roots.slice();
  for(const [, kids] of children) if(kids.length > 1) for(const k of kids) starts.push(k);
  const seen = new Uint8Array(N); const chains = [];
  for(const s of starts){
    if(seen[s]) continue;
    let chain = [s]; seen[s] = 1; let cur = s;
    for(;;){
      const kids = children.get(cur);
      if(!kids || kids.length !== 1 || seen[kids[0]]) break;
      cur = kids[0]; seen[cur] = 1; chain.push(cur);
    }
    const head = index.has(par[chain[0]]) ? index.get(par[chain[0]]) : -1;
    if(head >= 0 && head !== chain[0]) chain = [head].concat(chain);
    if(chain.length >= 2) chains.push(chain);
  }
  return chains;
}
function smoothInPlace(cols, passes, lam){    // render._smooth, per column (Jacobi update)
  const n = cols[0].length;
  if(n < 3 || passes <= 0) return;
  for(let it=0; it<passes; it++){
    for(const p of cols){
      const old = Float64Array.from(p);
      for(let i=1;i<n-1;i++) p[i] = old[i] + lam*(old[i-1] + old[i+1] - 2*old[i])*0.5;
    }
  }
}
function prepare(sk){
  if(sk._ps) return sk._ps;
  if(!sk.rs || !sk.ids) throw new Error("parallel-styles: skeleton lacks radii/ids");
  const X=sk.xs, Y=sk.ys, Z=sk.zs, Rr=sk.rs;
  let chains = swcChains(sk).map(c => c);
  // long-edge cut: max_jump=8 x median, jump_p99=4 x p99, measured on raw nodes
  const dAll = [];
  for(const c of chains) for(let k=1;k<c.length;k++){
    const a=c[k-1], b=c[k];
    dAll.push(Math.hypot(X[b]-X[a], Y[b]-Y[a], Z[b]-Z[a]));
  }
  if(dAll.length){
    const limit = Math.max(8.0*Math.max(median(dAll),1e-9), 4.0*Math.max(percentile(dAll,99),1e-9));
    const cut = [];
    for(const c of chains){
      let a = 0;
      for(let k=1;k<c.length;k++){
        const p=c[k-1], q=c[k];
        if(Math.hypot(X[q]-X[p], Y[q]-Y[p], Z[q]-Z[p]) > limit){
          if(k - a >= 2) cut.push(c.slice(a, k));
          a = k;
        }
      }
      if(a === 0) cut.push(c); else if(c.length - a >= 2) cut.push(c.slice(a));
    }
    chains = cut;
  }
  let total = 0; for(const c of chains) total += c.length-1;
  const segs = new Float64Array(total*6), rads = new Float64Array(total);
  let o = 0;
  for(const c of chains){
    const n = c.length;
    const px=new Float64Array(n), py=new Float64Array(n), pz=new Float64Array(n), pr=new Float64Array(n);
    for(let k=0;k<n;k++){ px[k]=X[c[k]]; py[k]=Y[c[k]]; pz[k]=Z[c[k]]; pr[k]=Rr[c[k]]; }
    smoothInPlace([px,py,pz], 4, 0.5);
    smoothInPlace([pr], 4, 0.5);
    for(let k=0;k<n-1;k++){
      segs[o*6]=px[k]; segs[o*6+1]=py[k]; segs[o*6+2]=pz[k];
      segs[o*6+3]=px[k+1]; segs[o*6+4]=py[k+1]; segs[o*6+5]=pz[k+1];
      rads[o] = (pr[k]+pr[k+1])*0.5; o++;
    }
  }
  sk._ps = {segs, rads, n: total};
  return sk._ps;
}

// frontal projection: (x, -y), depth = mean z of the segment
function project(ps){
  const n = ps.n, P = new Float64Array(n*4), depth = new Float64Array(n), S = ps.segs;
  for(let i=0;i<n;i++){
    P[i*4]=S[i*6]; P[i*4+1]=-S[i*6+1]; P[i*4+2]=S[i*6+3]; P[i*4+3]=-S[i*6+4];
    depth[i] = (S[i*6+2]+S[i*6+5])/2;
  }
  return {P, depth};
}

// --- matplotlib tick locator (AutoLocator = MaxNLocator steps 1,2,2.5,5,10) ---
function autoTicks(vmin, vmax, nbins){
  const dv = Math.abs(vmax-vmin), meanv = (vmax+vmin)/2;
  let offset = 0;
  if(!(Math.abs(meanv)/dv < 100)) offset = Math.sign(meanv) * Math.pow(10, Math.floor(Math.log10(Math.abs(meanv))));
  const scale = Math.pow(10, Math.floor(Math.log10(dv/nbins)));
  const _vmin = vmin-offset, _vmax = vmax-offset;
  const steps = [0.1,0.2,0.25,0.5,1,2,2.5,5,10,20].map(s=>s*scale);
  const raw = (_vmax-_vmin)/nbins;
  let istep = steps.findIndex(s => s >= raw); if(istep < 0) istep = steps.length-1;
  const close = (a,b) => Math.abs(a-b) < 1e-10;
  let ticks = [];
  for(let k=istep; k>=0; k--){
    const step = steps[k];
    const best = Math.floor(_vmin/step)*step;
    const le = x => { const d=Math.floor(x/step), m=x-d*step; return close(m/step,1)?d+1:d; };
    const ge = x => { const d=Math.floor(x/step), m=x-d*step; return close(m/step,0)?d:d+1; };
    const lo = le(_vmin-best), hi = ge(_vmax-best);
    ticks = []; for(let t=lo;t<=hi;t++) ticks.push(t*step+best);
    const nt = ticks.filter(t => t<=_vmax && t>=_vmin).length;
    if(nt >= 2) break;
  }
  return ticks.map(t=>t+offset).filter(t => t >= vmin - 1e-9*dv && t <= vmax + 1e-9*dv);
}

// --- drawing ------------------------------------------------------------------
const BG = {ink:"#f4f0e6", spectral:"#07080d", blueprint:"#0a1b2e", duotone:"#efe9df", constellation:"#0b0b0c"};
const FG = {ink:"#1b1815", spectral:"#c9d4e8", blueprint:"#7fe3ff", duotone:"#1c3f94", constellation:"#d9a441"};

function hexA(hex, a){
  const n = parseInt(hex.slice(1),16);
  return "rgba("+(n>>16&255)+","+(n>>8&255)+","+(n&255)+","+a+")";
}

function paint(cv, sk, style, opts){
  opts = opts || {};
  if(STYLES.indexOf(style) < 0) style = "ink";
  const paper = opts.paper || 12;
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  const SCALE = Math.hypot(paper, paper*H/W) / REF_DIAGONAL;
  const ptPx = W / (paper*72);                         // points -> canvas px
  const lwPx = v => Math.max(v*SCALE, MIN_LW_PT) * ptPx; // render._lw, in px

  ctx.save();
  ctx.setTransform(1,0,0,1,0,0);
  ctx.globalAlpha = 1;
  ctx.fillStyle = BG[style]; ctx.fillRect(0,0,W,H);

  const ps = prepare(sk);
  if(!ps.n){ ctx.restore(); return; }
  const {P, depth} = project(ps);

  // framing (portrait.py hero box, or render.py _fit_limits)
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(let i=0;i<P.length;i+=2){ const x=P[i], y=P[i+1];
    if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; }
  const rect = opts.rect || LAYOUT.rect;
  const frame = opts.frame || LAYOUT.frame;
  let bx0,by0,bx1,by1, pad;
  if(frame === "hero"){
    const cx=(x0+x1)/2, cy=(y0+y1)/2, half=Math.max(x1-x0, y1-y0)*LAYOUT.half;
    bx0=cx-half; bx1=cx+half; by0=cy-half; by1=cy+half; pad = 0;
  } else { bx0=x0; bx1=x1; by0=y0; by1=y1; pad = LAYOUT.pad; }
  let dw=Math.max(bx1-bx0,1e-9)*(1+2*pad), dh=Math.max(by1-by0,1e-9)*(1+2*pad);
  const target = (paper*W/W*rect[2]) / (paper*H/W*rect[3]);
  if(dw/dh < target) dw = dh*target; else dh = dw/target;
  const ccx=(bx0+bx1)/2, ccy=(by0+by1)/2;
  const xl0=ccx-dw/2, xl1=ccx+dw/2, yl0=ccy-dh/2, yl1=ccy+dh/2;
  const AX = rect[0]*W, AW = rect[2]*W, AY = H - (rect[1]+rect[3])*H, AH = rect[3]*H;
  const sx = AW/(xl1-xl0), sy = AH/(yl1-yl0);
  const X = x => AX + (x-xl0)*sx;
  const Y = y => AY + (yl1-y)*sy;

  const seg = (i, dx, dy) => {           // one 2-point segment as its own path
    ctx.beginPath();
    ctx.moveTo(X(P[i*4]+dx), Y(P[i*4+1]+dy));
    ctx.lineTo(X(P[i*4+2]+dx), Y(P[i*4+3]+dy));
    ctx.stroke();
  };
  const n = ps.n;

  // blueprint keeps its axes: grid under the drawing, spines + ticks on top
  let ticksX = null, ticksY = null;
  if(style === "blueprint"){
    const lab = 5*SCALE;                                     // _fs(5) pt
    const nbx = Math.max(Math.min(Math.floor(rect[2]*paper*72/(lab*3)), 9), 1);
    const nby = Math.max(Math.min(Math.floor(rect[3]*paper*72*H/W/(lab*2)), 9), 1);
    ticksX = autoTicks(xl0, xl1, nbx); ticksY = autoTicks(yl0, yl1, nby);
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(AX, AY, AW, AH); ctx.clip();     // matplotlib clips to the axes
  ctx.fillStyle = BG[style]; ctx.fillRect(AX, AY, AW, AH);

  if(style === "ink"){
    const w = norm(ps.rads);
    ctx.lineCap = "round"; ctx.strokeStyle = hexA("#1b1815", 0.85);
    for(let i=0;i<n;i++){ ctx.lineWidth = lwPx(0.15 + 0.9*w[i]); seg(i,0,0); }
  }
  else if(style === "spectral"){
    // neuron_sheet.py style_spectral_single: hue = distance from the root
    // (cool at the root, warm at the tips), full brightness, over a soft glow.
    const R = ps.rads; let k = 0; for(let i=1;i<n;i++) if(R[i] > R[k]) k = i;
    const rx = P[k*4], ry = P[k*4+1];                      // root = start of the thickest segment
    const w = norm(R), dist = new Float64Array(n);
    for(let i=0;i<n;i++) dist[i] = Math.hypot((P[i*4]+P[i*4+2])/2 - rx, (P[i*4+1]+P[i*4+3])/2 - ry);
    const t = norm(dist), rgb = new Array(n);
    for(let i=0;i<n;i++){
      const c = hsvToRgb(0.60 - 0.55*t[i], 0.70, 1.0);
      rgb[i] = Math.round(c[0]*255)+","+Math.round(c[1]*255)+","+Math.round(c[2]*255);
    }
    ctx.lineCap = "round";
    for(let i=0;i<n;i++){ ctx.strokeStyle = "rgba("+rgb[i]+",0.12)"; ctx.lineWidth = lwPx(0.2 + 1.1*w[i])*3.2; seg(i,0,0); }
    for(let i=0;i<n;i++){ ctx.strokeStyle = "rgba("+rgb[i]+",0.92)"; ctx.lineWidth = lwPx(0.2 + 1.1*w[i]); seg(i,0,0); }
  }
  else if(style === "blueprint"){
    // grid (axisbelow): #1e4a72, lw _lw(0.4), alpha 0.6
    ctx.lineCap = "butt";
    ctx.strokeStyle = hexA("#1e4a72", 0.6); ctx.lineWidth = lwPx(0.4);
    for(const t of ticksX){ const x=X(t); ctx.beginPath(); ctx.moveTo(x,AY); ctx.lineTo(x,AY+AH); ctx.stroke(); }
    for(const t of ticksY){ const y=Y(t); ctx.beginPath(); ctx.moveTo(AX,y); ctx.lineTo(AX+AW,y); ctx.stroke(); }
    // faint wide underlay, then the crisp hairline
    ctx.strokeStyle = hexA("#2ec5ff", 0.10); ctx.lineWidth = lwPx(1.6);
    for(let i=0;i<n;i++) seg(i,0,0);
    ctx.strokeStyle = hexA("#7fe3ff", 0.9); ctx.lineWidth = lwPx(0.35);
    for(let i=0;i<n;i++) seg(i,0,0);
  }
  else if(style === "duotone"){
    let px0=Infinity,px1=-Infinity,py0=Infinity,py1=-Infinity;
    for(let i=0;i<P.length;i+=2){ const x=P[i],y=P[i+1];
      if(x<px0)px0=x; if(x>px1)px1=x; if(y<py0)py0=y; if(y>py1)py1=y; }
    const ox=(px1-px0)*0.006, oy=(py1-py0)*0.006;
    ctx.lineCap = "butt"; ctx.lineWidth = lwPx(0.9);
    ctx.strokeStyle = hexA("#ff4f37", 0.55); for(let i=0;i<n;i++) seg(i, ox, oy);
    ctx.strokeStyle = hexA("#1c3f94", 0.55); for(let i=0;i<n;i++) seg(i, -ox, -oy);
  }
  else if(style === "constellation"){
    // neuron_sheet.py style_constellation_single: a gold star field. Every
    // skeleton point is a star sized by thickness; branch tips are brighter;
    // the root is a white star.
    const R = ps.rads; let k = 0; for(let i=1;i<n;i++) if(R[i] > R[k]) k = i;
    const rx = P[k*4], ry = P[k*4+1];
    const m = n*2, rr = new Float64Array(m);
    for(let i=0;i<n;i++){ rr[2*i] = R[i]; rr[2*i+1] = R[i]; }
    const sz = norm(rr);
    const dotPx = r => Math.sqrt(r)*SCALE*ptPx/2;            // scatter s = area in pt^2 -> radius px
    const star = (x, y, r) => { ctx.beginPath(); ctx.arc(X(x), Y(y), r, 0, Math.PI*2); ctx.fill(); };
    ctx.fillStyle = hexA("#f3d79a", 0.55);
    for(let j=0;j<m;j++) star(P[j*2], P[j*2+1], dotPx(0.5 + 5.5*sz[j]));
    // tips: endpoints (rounded to 0.01) that only one segment reaches
    const cnt = new Map();
    for(let j=0;j<m;j++){ const key = Math.round(P[j*2]*100)+","+Math.round(P[j*2+1]*100); cnt.set(key, (cnt.get(key)||0)+1); }
    ctx.fillStyle = hexA("#ffe9b0", 0.95);
    const rt = dotPx(6);
    for(const [key, c] of cnt){ if(c !== 1) continue; const q = key.split(","); star(+q[0]/100, +q[1]/100, rt); }
    ctx.fillStyle = "#ffffff";
    star(rx, ry, dotPx(30));
  }
  ctx.restore();

  if(style === "blueprint"){
    // spines (#1e4a72, default 0.8pt), ticks out (#4e88b8, _lw(0.5), 3*SCALE pt), labels _fs(5)
    ctx.strokeStyle = "#1e4a72"; ctx.lineWidth = 0.8*ptPx; ctx.lineCap = "butt";
    ctx.strokeRect(AX, AY, AW, AH);
    ctx.strokeStyle = "#4e88b8"; ctx.lineWidth = lwPx(0.5);
    const tl = 3*SCALE*ptPx, padPx = 3.5*ptPx;
    ctx.fillStyle = "#4e88b8";
    ctx.font = (5*SCALE*ptPx).toFixed(2)+"px 'DejaVu Sans', Verdana, sans-serif";
    const lbl = v => { const r = Math.round(v*1e6)/1e6; return (r<0?"−":"") + String(Math.abs(r)); };
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for(const t of ticksX){ const x=X(t); ctx.beginPath(); ctx.moveTo(x,AY+AH); ctx.lineTo(x,AY+AH+tl); ctx.stroke();
      ctx.fillText(lbl(t), x, AY+AH+tl+padPx); }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for(const t of ticksY){ const y=Y(t); ctx.beginPath(); ctx.moveTo(AX,y); ctx.lineTo(AX-tl,y); ctx.stroke();
      ctx.fillText(lbl(t), AX-tl-padPx, y); }
  }

  if(opts.label){
    // neuron_sheet.py fig.text calls: (x, baseline y, size pt, alpha, align)
    const fg = FG[style], L = opts.label;
    const txt = (s, x, y, pt, a, align, weight) => {
      if(!s) return;
      ctx.globalAlpha = a; ctx.textAlign = align;
      ctx.font = (weight||"400")+" "+(pt*SCALE*ptPx).toFixed(2)+"px 'DejaVu Sans', Verdana, sans-serif";
      ctx.fillText(s, x*W, H*(1-y));
    };
    ctx.fillStyle = fg; ctx.textBaseline = "alphabetic";
    txt(String(L.type||""), 0.08, 0.935, 27, 1, "left", "300");
    txt(String(L.subtitle||"").toUpperCase(), 0.08, 0.906, 7.5, 0.75, "left");
    txt("bodyId "+L.bodyId, 0.08, 0.068, 7, 0.7, "left");
    txt("MaleCNS v1.0  ·  frontal view", 0.08, 0.050, 5.5, 0.5, "left");
    txt(CREDIT, 0.92, 0.030, 4.3, 0.55, "right");
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

// Small canvases: draw the sheet at print-preview resolution off screen and
// scale it down, so hairlines and small type land the way a downsized print
// does (drawing sub-pixel strokes directly comes out visibly heavier).
const SS_MIN = 1800;                     // = the sheet at 150 dpi
let _off = null;
function drawSheet(cv, sk, style, opts){
  opts = opts || {};
  const W = cv.width, H = cv.height;
  if(opts.direct || W >= SS_MIN || typeof document === "undefined"){ paint(cv, sk, style, opts); return; }
  let k = 1; while(W*k < SS_MIN) k *= 2;
  if(!_off) _off = document.createElement("canvas");
  _off.width = W*k; _off.height = H*k;
  paint(_off, sk, style, opts);
  let src = _off, w = W*k, h = H*k;
  while(w > W){                           // halve step by step (box filter each time)
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
}

window.ParallelStyles = {STYLES, BG, FG, LAYOUT, prepare, drawSheet, subtitleFor,
  _internal:{percentile, norm, swcChains, project, autoTicks}};
})();
