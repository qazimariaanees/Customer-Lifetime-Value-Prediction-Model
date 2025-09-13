# Customer-Lifetime-Value-Prediction-Model
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Customer Spend Prediction — Interactive Demo</title>
  <!-- Google Fonts -->
  <link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Playfair+Display:wght@600;700&display=swap' rel='stylesheet'>
  <!-- Plotly for interactive charts -->
  <script src='https://cdn.plot.ly/plotly-2.24.1.min.js'></script>
  <style>
    :root{
      --bg:#07070a; --card:#0f1115; --accent:#ff6b6b; --muted:#9aa1a8; --glass: rgba(255,255,255,0.03);
    }
    html,body{height:100%;margin:0;background:linear-gradient(180deg,#000000 0%, #071020 55%);color:#e6eef6;font-family:Inter, system-ui, sans-serif}
    .wrap{max-width:1150px;margin:28px auto;padding:24px}
    header{display:flex;align-items:center;gap:18px}
    .logo{font-family:'Playfair Display',serif;font-size:22px;font-weight:700;color:var(--accent);text-shadow:0 6px 18px rgba(255,107,107,0.08)}
    .subtitle{color:var(--muted);font-size:14px}

    .grid{display:grid;grid-template-columns:360px 1fr;gap:20px;margin-top:20px}
    .panel{background:var(--card);border-radius:12px;padding:18px;box-shadow:0 6px 30px rgba(3,6,10,0.6);backdrop-filter:blur(6px)}

    label{display:block;font-size:13px;color:#cfe3ff;margin-bottom:6px}
    .control{margin-bottom:16px}
    input[type=range]{width:100%}

    .muted{color:var(--muted);font-size:13px}
    .big{font-size:20px;font-weight:700}

    .btn{display:inline-block;padding:8px 12px;border-radius:8px;background:linear-gradient(90deg,var(--accent),#ffa66b);color:#061217;font-weight:700;border:none;cursor:pointer;box-shadow:0 6px 18px rgba(255,107,107,0.12);}
    .btn-ghost{background:transparent;border:1px solid rgba(255,255,255,0.04);color:#e6eef6}

    #chart{height:520px}
    .mini-charts{display:flex;gap:14px;margin-top:14px}
    .mini{flex:1;padding:12px;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border-radius:10px}

    .glow{color:var(--accent);text-shadow:0 4px 18px rgba(255,107,107,0.18)}

    .footer-note{margin-top:10px;color:var(--muted);font-size:13px}

    /* small animation */
    .pulse{animation:pulse 3s ease-in-out infinite}
    @keyframes pulse{0%{transform:translateY(0)}50%{transform:translateY(-4px)}100%{transform:translateY(0)}}

    pre{white-space:pre-wrap;font-size:13px;color:#d8e8ff}
    .download{display:flex;justify-content:space-between;align-items:center}

    /* responsive */
    @media(max-width:880px){.grid{grid-template-columns:1fr;}.logo{font-size:18px}}
  </style>
</head>
<body>
  <div class='wrap'>
    <header>
      <div class='logo'>Customer Spend Prediction</div>
      <div>
        <div class='subtitle'>Interactive explainer + segmentation demo — target under-performing high-potential customers</div>
        <div class='muted'>Upload your CSV (or use sample) → filter → download segment</div>
      </div>
    </header>

    <div class='grid'>
      <!-- Controls -->
      <div class='panel'>
        <h3 style='margin:0 0 8px 0'>Controls</h3>
        <p class='muted'>Use the controls below to filter and create a targeted segment of customers.</p>

        <div class='control'>
          <label class='muted'>Upload transactions CSV (columns: customer_id, frequency, pred_prob, spend_90_total, pred_spend, spend_actual_vs_pred, shortfall, recency, monetary)</label>
          <input id='file' type='file' accept='text/csv' />
        </div>

        <div class='control'>
          <label>Min propensity (pred probability)</label>
          <input id='minProb' type='range' min='0' max='1' step='0.01' value='0.4' />
          <div class='muted'>Current: <span id='minProbVal'>0.40</span></div>
        </div>

        <div class='control'>
          <label>Shortfall threshold (₹)</label>
          <input id='shortfall' type='range' min='0' max='10000' step='50' value='500' />
          <div class='muted'>Current: <span id='shortfallVal'>₹500</span></div>
        </div>

        <div class='control download'>
          <div>
            <div class='muted'>Segment size</div>
            <div class='big' id='segmentSize'>0</div>
          </div>
          <div>
            <button id='downloadBtn' class='btn'>Download Segmentation</button>
          </div>
        </div>

        <hr style='border:none;border-top:1px solid rgba(255,255,255,0.04);margin:14px 0'>

        <div class='muted'>Quick actions</div>
        <div style='display:flex;gap:8px;margin-top:8px'>
          <button id='useSample' class='btn-ghost'>Use sample data</button>
          <button id='reset' class='btn-ghost'>Reset filters</button>
        </div>

        <p class='footer-note'>Tip: Increase <em>Min propensity</em> to focus only on customers the model was confident about. Increase <em>Shortfall</em> to only keep big misses.</p>
      </div>

      <!-- Main chart -->
      <div class='panel'>
        <div style='display:flex;justify-content:space-between;align-items:center'>
          <div>
            <h2 style='margin:6px 0'>Interactive scatter — <span class='glow'>Visualize & segment</span></h2>
            <div class='muted'>Dots: customers. X = frequency. Y = pred_prob. Color = spent vs predicted.</div>
          </div>
          <div class='muted'>Made with ❤️ & Plotly</div>
        </div>

        <div id='chart'></div>

        <div class='mini-charts'>
          <div class='mini'>
            <div class='muted'>Calibration Curve</div>
            <div id='calib' style='height:140px'></div>
          </div>
          <div class='mini'>
            <div class='muted'>Brier Score</div>
            <div style='display:flex;flex-direction:column;align-items:flex-start;justify-content:center;height:140px'>
              <div class='big glow pulse' id='brierVal'>—</div>
              <div class='muted'>Lower is better (0 = perfect)</div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <section style='margin-top:18px' class='panel'>
      <h3 style='margin-top:0'>About this interactive repo (what you'll get)</h3>
      <ul>
        <li><strong>Upload</strong> your CSV of customer-level features (or use the sample provided).</li>
        <li><strong>Visualize</strong> how predictions vs actuals behave and spot underperformers (red dots).</li>
        <li><strong>Filter</strong> by predicted propensity and shortfall amount to carve a marketing segment.</li>
        <li><strong>Download</strong> the filtered segment as a CSV — ready for email/SMS campaigns.</li>
      </ul>
      <p class='muted'>This single-file demo works offline in your browser (except for the Plotly CDN). Copy this file into your GitHub repo as <code>index.html</code>, and add a short README (included inside the file as comments) to make the repository presentable.</p>
    </section>

  </div>

  <script>
  // Sample data (fallback)
  const sampleData = [
    {customer_id:'C001', frequency:8, pred_prob:0.82, spend_90_total:1200, pred_spend:1500, spend_actual_vs_pred:-300, shortfall:300, recency:12, monetary:1200},
    {customer_id:'C002', frequency:3, pred_prob:0.65, spend_90_total:500, pred_spend:700, spend_actual_vs_pred:-200, shortfall:200, recency:40, monetary:500},
    {customer_id:'C003', frequency:15, pred_prob:0.95, spend_90_total:3800, pred_spend:3500, spend_actual_vs_pred:300, shortfall:0, recency:5, monetary:3800},
    {customer_id:'C004', frequency:6, pred_prob:0.51, spend_90_total:80, pred_spend:800, spend_actual_vs_pred:-720, shortfall:720, recency:90, monetary:80},
    {customer_id:'C005', frequency:2, pred_prob:0.30, spend_90_total:0, pred_spend:400, spend_actual_vs_pred:-400, shortfall:400, recency:120, monetary:0},
    {customer_id:'C006', frequency:10, pred_prob:0.78, spend_90_total:1400, pred_spend:1800, spend_actual_vs_pred:-400, shortfall:400, recency:18, monetary:1400},
  ];

  let data = sampleData.slice();

  // Helpers
  const el = id => document.getElementById(id);
  el('minProb').addEventListener('input', e=>{el('minProbVal').innerText = Number(e.target.value).toFixed(2); applyFilters()});
  el('shortfall').addEventListener('input', e=>{el('shortfallVal').innerText = '₹'+Number(e.target.value); applyFilters()});
  el('useSample').addEventListener('click', ()=>{data = sampleData.slice(); renderAll();});
  el('reset').addEventListener('click', ()=>{el('minProb').value=0.4;el('shortfall').value=500;el('minProbVal').innerText='0.40';el('shortfallVal').innerText='₹500';data = sampleData.slice(); renderAll();});
  el('downloadBtn').addEventListener('click', downloadSegment);

  // File upload -> parse CSV (simple split, works for clean CSVs)
  el('file').addEventListener('change', e=>{
    const f = e.target.files[0];
    if(!f) return;
    const reader = new FileReader();
    reader.onload = ev => {
      const text = ev.target.result;
      data = csvToObjects(text);
      renderAll();
    };
    reader.readAsText(f);
  });

  function csvToObjects(text){
    const lines = text.trim().split('\n');
    const headers = lines.shift().split(',').map(h=>h.trim());
    return lines.map(line=>{
      const values = line.split(',').map(v=>v.trim());
      const obj = {};
      headers.forEach((h,i)=>{
        const val = values[i] || '';
        if(['frequency','pred_prob','spend_90_total','pred_spend','spend_actual_vs_pred','shortfall','recency','monetary'].includes(h)){
          obj[h]= Number(val);
        } else obj[h]= val;
      });
      return obj;
    });
  }

  function applyFilters(){
    const minP = Number(el('minProb').value);
    const shortT = Number(el('shortfall').value);
    const filtered = data.filter(d => (Number(d.pred_prob) >= minP) && (Number(d.shortfall || 0) >= shortT));
    el('segmentSize').innerText = filtered.length;
    return filtered;
  }

  function renderScatter(){
    const x = data.map(d=>d.frequency);
    const y = data.map(d=>d.pred_prob);
    const text = data.map(d=>`ID: ${d.customer_id}<br>freq: ${d.frequency}<br>pred: ${d.pred_prob}<br>shortfall: ₹${d.shortfall || 0}`);
    const colors = data.map(d=>Number(d.shortfall||0) > 0 ? 'rgb(255,90,90)' : (Number(d.spend_actual_vs_pred||0) > 0 ? 'rgb(80,170,255)' : 'rgb(220,200,60)'));

    const layout = {margin:{t:30,l:40,r:20,b:40},xaxis:{title:'Frequency', gridcolor:'rgba(255,255,255,0.03)'},yaxis:{title:'Predicted propensity', range:[0,1], gridcolor:'rgba(255,255,255,0.03)'},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'};
    Plotly.newPlot('chart', [{x:x,y:y,text:text,mode:'markers',marker:{size:12,color:colors,opacity:0.9}}], layout, {responsive:true});
  }

  function renderCalibration(){
    const bins = 5; const counts = Array(bins).fill(0); const succ = Array(bins).fill(0);
    data.forEach(d=>{
      const p = Number(d.pred_prob||0); const idx = Math.min(bins-1, Math.floor(p*bins));
      counts[idx] += 1; if(Number(d.spend_90_total) > 0) succ[idx] += 1;
    });
    const x = []; const y = [];
    for(let i=0;i<bins;i++){ x.push((i+0.5)/bins); y.push(counts[i]===0? null : (succ[i]/counts[i])); }
    const layout = {margin:{t:10,l:30,r:10,b:30},xaxis:{title:'Predicted'},yaxis:{title:'Observed',range:[0,1]},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'};
    const trace1 = {x:x, y:y, mode:'lines+markers', name:'Model', marker:{size:6}};
    const trace2 = {x:[0,1], y:[0,1], mode:'lines', line:{dash:'dash'}, name:'Perfect'};
    Plotly.newPlot('calib',[trace1,trace2], layout, {displayModeBar:false,responsive:true});
  }

  function calcBrier(){
    const n = data.length; if(n===0) return null;
    let sum = 0; data.forEach(d=>{ const p = Number(d.pred_prob||0); const o = Number(d.spend_90_total>0); sum += (p - o)*(p - o); });
    return sum / n;
  }

  function renderAll(){ renderScatter(); renderCalibration(); const b = calcBrier(); el('brierVal').innerText = b===null? '—' : b.toFixed(3); el('segmentSize').innerText = applyFilters().length; }

  function downloadSegment(){
    const seg = applyFilters(); if(seg.length===0){ alert('No customers match the filters.'); return; }
    const header = Object.keys(seg[0]);
    const rows = seg.map(r=> header.map(h=> '"'+(r[h]===undefined?'':r[h])+'"').join(','));
    const csv = [header.join(',')].concat(rows).join('\n');
    const blob = new Blob([csv],{type:'text/csv'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href=url; a.download='segment.csv'; a.click(); URL.revokeObjectURL(url);
  }

  // init
  renderAll();
  </script>

  <!-- README (copy into your repo as README.md) -->
  <!--
  README: Customer Spend Prediction — Interactive single-file demo

  What this repo contains
  - index.html : single-file interactive demo that visualizes customer-level predictions and actual spend.

  How to use
  1. Open index.html in your browser (double-click or host via GitHub Pages) — Plotly is loaded via CDN.
  2. Upload a CSV with these columns (recommended):
     customer_id,frequency,pred_prob,spend_90_total,pred_spend,spend_actual_vs_pred,shortfall,recency,monetary
  3. Use the sliders to set minimum predicted probability and shortfall threshold.
  4. Click "Download Segmentation" to get the CSV of customers matching the filter.

  Suggestions for GitHub repo
  - Add a LICENSE (MIT if you want permissive).
  - Add a screenshot or GIF in the README for presentation.
  - Optionally split styles and JS into separate files (styles.css, script.js) if you prefer.

  -->
</body>
</html>
