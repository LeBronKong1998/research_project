'use strict';
(() => {
  const $ = id => document.getElementById(id);
  try {
    const meta = JSON.parse($('trajectory-data').textContent);
    const bytes = Uint8Array.from(atob(meta.csv_base64), c => c.charCodeAt(0));
    function parseCSV(text) {
      const records = []; let row = [], field = '', quoted = false;
      for (let i = 0; i < text.length; i++) {
        const c = text[i];
        if (quoted) {
          if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
          else if (c === '"') quoted = false; else field += c;
        } else if (c === '"') quoted = true;
        else if (c === ',') { row.push(field); field = ''; }
        else if (c === '\n' || c === '\r') {
          if (c === '\r' && text[i + 1] === '\n') i++;
          row.push(field); if (row.some(v => v !== '')) records.push(row); row = []; field = '';
        } else field += c;
      }
      if (field || row.length) { row.push(field); records.push(row); }
      if (quoted) throw Error('CSV 引号未闭合');
      return records;
    }
    const parsed = parseCSV(new TextDecoder().decode(bytes).replace(/^\uFEFF/, ''));
    const headers = parsed.shift(), cases = new Map();
    for (const values of parsed) {
      if (values.length !== headers.length) throw Error('CSV 列数不匹配');
      const raw = Object.fromEntries(headers.map((h, i) => [h, values[i]]));
      const r = { raw, values, case_id: raw.case_id, track_id: raw.track_id };
      for (const k of ['frame_id', 'timestamp_ms', 'x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width']) r[k] = raw[k] === '' ? null : Number(raw[k]);
      if (!cases.has(r.case_id)) cases.set(r.case_id, []);
      cases.get(r.case_id).push(r);
    }
    const allIds = [...cases.keys()].sort((a, b) => Number(a) - Number(b));
    let mode = 'raw', activeSample = null, availableIds = allIds, scene, tracks, frames, frameIndex = 0;
    let selectedTrack = '', timer = null, rngState = 7, bounds, hitTargets = [];
    const canvas = $('trajectory-canvas'), ctx = canvas.getContext('2d');
    const nfmt = n => Number(n).toFixed(3);
    function option(select, value, text) { const o = document.createElement('option'); o.value = value; o.textContent = text; select.append(o); }
    $('stat-rows').textContent = meta.row_count.toLocaleString(); $('stat-cases').textContent = allIds.length; $('stat-tracks').textContent = meta.track_count;
    $('official-license').textContent = meta.official_license;
    $('source-filename').textContent = meta.filename; $('source-sha').textContent = meta.sha256; $('source-link').href = meta.source_url;
    $('map-source').textContent = `地图 ${meta.map.filename} · ${meta.map.node_count} 节点 / ${meta.map.way_count} 线要素 · 官方 UTM ${meta.map.projection.utm_zone} 带，原点 (0,0)；与发布方 OSM_XY 最大差 ${meta.map.osm_xy_max_error_m.toExponential(2)} m。地图仅用于可视化。`;
    for (const o of $('view-mode').options) if (o.value !== 'raw') o.disabled = !meta.reviews[o.value];
    const head = document.createElement('tr');
    for (const h of headers) { const th = document.createElement('th'); th.textContent = h; th.scope = 'col'; head.append(th); }
    $('raw-head').append(head);
    function stop() { if (timer !== null) clearTimeout(timer); timer = null; $('play').textContent = '播放'; $('play').setAttribute('aria-label', '播放轨迹'); }
    function schedule() { timer = setTimeout(() => { frameIndex = (frameIndex + 1) % frames.length; render(); schedule(); }, 100 / Number($('speed').value)); }
    function play() { if (timer !== null) return stop(); $('play').textContent = '暂停'; $('play').setAttribute('aria-label', '暂停轨迹'); schedule(); }
    function random() { rngState = (Math.imul(rngState, 1664525) + 1013904223) >>> 0; return rngState / 4294967296; }
    function selectCase(id, sample = null) {
      stop(); $('case-select').value = id; scene = cases.get(id); tracks = new Map();
      for (const r of scene) { if (!tracks.has(r.track_id)) tracks.set(r.track_id, []); tracks.get(r.track_id).push(r); }
      for (const rows of tracks.values()) rows.sort((a, b) => a.frame_id - b.frame_id);
      frames = [...new Set(scene.map(r => r.frame_id))].sort((a, b) => a - b);
      if (mode !== 'raw') {
        const eligible = meta.reviews[mode].samples.filter(s => s.case_id === id);
        activeSample = sample || eligible[0];
        $('sample-select').replaceChildren();
        for (const s of eligible) option($('sample-select'), String(s.sample_index), `车辆 ${s.track_id} · 样本 ${s.sample_index}`);
        $('sample-select').value = String(activeSample.sample_index);
      } else activeSample = null;
      const travel = rows => Math.hypot(rows.at(-1).x - rows[0].x, rows.at(-1).y - rows[0].y);
      selectedTrack = activeSample ? activeSample.track_id : [...tracks].sort((a, b) => Number(b[1].length === frames.length) - Number(a[1].length === frames.length) || travel(b[1]) - travel(a[1]))[0][0];
      $('track-select').replaceChildren();
      for (const [tid, rows] of [...tracks].sort((a, b) => Number(a[0]) - Number(b[0]))) option($('track-select'), tid, `${rows[0].raw.agent_type === 'car' ? '车辆' : '行人/自行车'} ${tid} · ${rows.length} 帧`);
      // A prediction review always stays attached to its actual model target.
      $('track-select').disabled = mode !== 'raw'; $('track-select').value = selectedTrack;
      frameIndex = activeSample ? frames.indexOf(activeSample.history_frames.at(-1)) : 0;
      $('frame-slider').max = frames.length - 1; resetBounds(); render(); renderPredictionTable();
    }
    function setMode(value) {
      mode = value; $('view-mode').value = value;
      const review = meta.reviews[mode], model = mode !== 'raw';
      for (const id of ['model-summary', 'sample-label', 'prediction-label', 'sample-metrics', 'prediction-table-card']) $(id).hidden = !model;
      $('mode-badge').textContent = model ? `${mode === 'val' ? '验证集' : '测试集'} · 真实模型输出` : '真实观测数据 · 无模型预测';
      availableIds = model ? [...new Set(review.samples.map(s => s.case_id))].sort((a, b) => Number(a) - Number(b)) : allIds;
      $('case-select').replaceChildren(); for (const id of availableIds) option($('case-select'), id, `Case ${id}`);
      if (model) {
        $('summary-ade').textContent = nfmt(review.metrics.top1ADE); $('summary-fde').textContent = nfmt(review.metrics.top1FDE);
        $('summary-cv').textContent = `${nfmt(review.constant_velocity.top1ADE)} / ${nfmt(review.constant_velocity.top1FDE)}`;
        $('summary-count').textContent = review.sample_count;
        $('summary-count-label').textContent = review.summary.subset_evaluation ? '本次评估子集目标（非完整集合）' : '当前集合 · 完整评估目标';
        if (review.training_max_batches) $('mode-badge').textContent += ' · 截断训练权重';
        $('checkpoint-source').textContent = `权重 ${review.checkpoint_filename} · 验证选中 epoch ${review.selected_epoch ?? '—'} · SHA-256 ${review.checkpoint_sha256} · ${mode} 集，仅使用历史输入；当前为自定义 case 划分。`;
      } else $('checkpoint-source').textContent = '';
      $('loss-card').hidden = !model || !review.training_history.length;
      selectCase(availableIds[Math.floor(random() * availableIds.length)]);
      drawLoss();
    }
    function resetBounds() {
      const b = meta.map.bounds; let minX = b[0], minY = b[1], maxX = b[2], maxY = b[3];
      const points = scene.map(r => [r.x, r.y]);
      if (activeSample) points.push(...activeSample.prediction_xy.flat());
      for (const [x, y] of points) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
      bounds = { minX: minX - 5, minY: minY - 5, maxX: maxX + 5, maxY: maxY + 5 };
    }
    function zoom(factor) {
      const cx = (bounds.minX + bounds.maxX) / 2, cy = (bounds.minY + bounds.maxY) / 2;
      const w = (bounds.maxX - bounds.minX) * factor, h = (bounds.maxY - bounds.minY) * factor;
      if (Math.min(w, h) < 5 || Math.max(w, h) > 1000) return;
      bounds = { minX: cx - w / 2, maxX: cx + w / 2, minY: cy - h / 2, maxY: cy + h / 2 }; draw();
    }
    function topMode() { return activeSample.probabilities.indexOf(Math.max(...activeSample.probabilities)); }
    function draw() {
      const rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width, h = rect.height, pad = { left: 62, right: 24, top: 22, bottom: 45 }, iw = w - pad.left - pad.right, ih = h - pad.top - pad.bottom;
      const scale = Math.min(iw / (bounds.maxX - bounds.minX), ih / (bounds.maxY - bounds.minY));
      const cx = (bounds.minX + bounds.maxX) / 2, cy = (bounds.minY + bounds.maxY) / 2;
      const px = x => pad.left + iw / 2 + (x - cx) * scale, py = y => pad.top + ih / 2 - (y - cy) * scale;
      ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#fcfdfc'; ctx.fillRect(0, 0, w, h);
      ctx.save(); ctx.beginPath(); ctx.rect(pad.left, pad.top, iw, ih); ctx.clip();
      if ($('show-map').checked) {
        // Same way types/colors/widths as official map_vis_without_lanelet.py.
        ctx.fillStyle = '#d3d3d3'; ctx.fillRect(pad.left, pad.top, iw, ih);
        const black = ['curbstone', 'road_border', 'guard_rail'];
        for (const way of meta.map.ways) {
          let color = '#fff', width = 1, dash = [];
          if (black.includes(way.type)) color = '#182b32';
          else if (way.type === 'line_thin' || way.type === 'line_thick') { width = way.type === 'line_thick' ? 2 : 1; if (way.subtype === 'dashed') dash = [10, 10]; }
          else if (['pedestrian_marking', 'bike_marking'].includes(way.type)) dash = [5, 10];
          else if (way.type === 'stop_line') width = 3;
          else if (way.type === 'virtual') { color = '#476cbe'; dash = [2, 5]; }
          else continue;
          path(way.points, color, dash, width);
        }
      }
      function path(points, color, dash = [], width = 1.5) {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath();
        points.forEach((p, i) => i ? ctx.lineTo(px(p[0]), py(p[1])) : ctx.moveTo(px(p[0]), py(p[1]))); ctx.stroke(); ctx.setLineDash([]);
      }
      function trackPath(rows, color, width = 1) { let run = [], previous; for (const r of rows) { if (previous && r.frame_id !== previous.frame_id + 1) { path(run, color, [], width); run = []; } run.push([r.x, r.y]); previous = r; } path(run, color, [], width); }
      if ($('show-others').checked) for (const [id, rows] of tracks) if (id !== selectedTrack) trackPath(rows, '#7a919d', 1);
      const frame = frames[frameIndex], chosen = tracks.get(selectedTrack), cutoff = activeSample ? activeSample.history_frames.at(-1) : frames[0] + 9;
      trackPath(chosen.filter(r => r.frame_id <= cutoff), '#1764c0', 3);
      trackPath(chosen.filter(r => r.frame_id >= cutoff), '#168253', 3);
      if (activeSample) {
        const top = topMode(), colors = ['#ce2862', '#834cb7', '#c76b08'];
        activeSample.prediction_xy.forEach((points, k) => {
          if ($('prediction-choice').value === 'top1' && k !== top) return;
          path([activeSample.history_xy.at(-1), ...points], k === top ? '#ce2862' : colors[k % colors.length], [7, 3], k === top ? 3 : 1.5);
          const j = activeSample.future_frames.indexOf(frame);
          if (j >= 0) {
            const pos = points[j]; ctx.fillStyle = k === top ? '#ce2862' : colors[k % colors.length]; ctx.beginPath(); ctx.arc(px(pos[0]), py(pos[1]), k === top ? 5 : 3, 0, Math.PI * 2); ctx.fill();
            if (k === top && $('show-errors').checked) path([pos, activeSample.truth_xy[j]], '#ce2862', [2, 2], 1.5);
          }
        });
      }
      hitTargets = [];
      for (const [id, rows] of tracks) {
        const r = rows.find(r => r.frame_id === frame); if (!r) continue;
        const active = id === selectedTrack; ctx.save(); ctx.translate(px(r.x), py(r.y)); ctx.rotate(-(r.psi_rad ?? 0));
        ctx.fillStyle = active ? '#1764c0' : '#405e6b'; ctx.strokeStyle = active ? '#0c4184' : '#fff'; ctx.lineWidth = active ? 1.8 : 1;
        if (r.length === null || r.width === null) { ctx.beginPath(); ctx.arc(0, 0, 3, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
        else { const len = r.length * scale, width = r.width * scale; ctx.fillRect(-len / 2, -width / 2, len, width); ctx.strokeRect(-len / 2, -width / 2, len, width); ctx.beginPath(); ctx.moveTo(len / 2, 0); ctx.lineTo(len / 2 - 3, -2); ctx.lineTo(len / 2 - 3, 2); ctx.closePath(); ctx.fillStyle = '#fff'; ctx.fill(); }
        ctx.restore(); hitTargets.push({ x: px(r.x), y: py(r.y), r });
        if ($('show-labels').checked) { ctx.font = active ? 'bold 12px system-ui' : '11px system-ui'; ctx.textAlign = 'left'; ctx.fillStyle = active ? '#104c96' : '#294859'; ctx.fillText(id, px(r.x) + 5, py(r.y) - 7); }
      }
      ctx.restore();
      const minX = cx - iw / (2 * scale), maxX = cx + iw / (2 * scale), minY = cy - ih / (2 * scale), maxY = cy + ih / (2 * scale);
      let tick = 10; while ((maxX - minX) / tick > 10) tick *= 2; if ((maxX - minX) / tick < 4) tick /= 2;
      ctx.font = '11px system-ui'; ctx.fillStyle = '#607685'; ctx.textAlign = 'center';
      for (let x = Math.ceil(minX / tick) * tick; x <= maxX; x += tick) ctx.fillText(x.toFixed(0), px(x), h - pad.bottom + 18);
      ctx.textAlign = 'right'; for (let y = Math.ceil(minY / tick) * tick; y <= maxY; y += tick) ctx.fillText(y.toFixed(0), pad.left - 8, py(y) + 4);
      ctx.textAlign = 'center'; ctx.fillText('世界坐标 x（m）', pad.left + iw / 2, h - 5);
      ctx.save(); ctx.translate(15, pad.top + ih / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('世界坐标 y（m）', 0, 0); ctx.restore();
      canvas.setAttribute('aria-label', `官方地图，Case ${$('case-select').value}，帧 ${frame}，目标 ${selectedTrack}，${mode === 'raw' ? '原始观测' : mode + ' 模型预测'}。`);
    }
    function render() {
      const frame = frames[frameIndex], rows = tracks.get(selectedTrack), r = rows.find(x => x.frame_id === frame);
      $('frame-slider').value = frameIndex; const stamp = scene.find(x => x.frame_id === frame).timestamp_ms;
      $('time-label').textContent = `${(stamp / 1000).toFixed(1)} s · 帧 ${frame}`;
      $('detail-type').textContent = (r ?? rows[0]).raw.agent_type; $('detail-frame').textContent = frame;
      $('detail-position').textContent = r ? `${r.x.toFixed(2)} / ${r.y.toFixed(2)}` : '本帧无记录'; $('detail-speed').textContent = r ? Math.hypot(r.vx, r.vy).toFixed(2) : '—';
      $('detail-heading').textContent = r?.psi_rad != null ? r.psi_rad.toFixed(3) : '原始值缺失'; $('detail-size').textContent = r?.length != null && r?.width != null ? `${r.length.toFixed(2)} × ${r.width.toFixed(2)}` : '原始值缺失'; $('detail-count').textContent = `${rows.length} 帧`;
      $('scene-hint').textContent = `Case ${$('case-select').value} · ${tracks.size} 条参与者轨迹 · ${scene.length.toLocaleString()} 行 · ${frames.length} 帧${activeSample ? ' · 预测从帧 ' + activeSample.future_frames[0] + ' 开始，未来真值仅用于对照' : ''}`;
      $('status').textContent = `${mode} · Case ${$('case-select').value} · 目标 ${selectedTrack}`;
      if (activeSample) {
        const m = activeSample.metrics; $('sample-top1').textContent = `${nfmt(m.top1ADE)} / ${nfmt(m.top1FDE)}`; $('sample-min').textContent = `${nfmt(m.minADE)} / ${nfmt(m.minFDE)}`;
        const j = activeSample.future_frames.indexOf(frame); $('sample-step-error').textContent = j >= 0 ? nfmt(Math.hypot(...activeSample.prediction_xy[topMode()][j].map((v, k) => v - activeSample.truth_xy[j][k]))) : '历史观测阶段';
      }
      draw(); renderTable();
    }
    function renderTable() {
      const frame = frames[frameIndex]; const records = $('table-mode').value === 'track' ? tracks.get(selectedTrack) : scene.filter(r => r.frame_id === frame).sort((a, b) => Number(a.track_id) - Number(b.track_id));
      $('table-caption').textContent = `${records.length} 行 · 原始 CSV 精度 · 点击定位`;
      $('raw-body').replaceChildren();
      for (const r of records) {
        const tr = document.createElement('tr'); tr.tabIndex = 0; if (r.track_id === selectedTrack && r.frame_id === frame) tr.className = 'selected';
        for (const v of r.values) { const td = document.createElement('td'); td.textContent = v; tr.append(td); }
        const select = () => { if (mode === 'raw') { selectedTrack = r.track_id; $('track-select').value = selectedTrack; } frameIndex = frames.indexOf(r.frame_id); render(); };
        tr.onclick = select; tr.onkeydown = e => { if (e.key === 'Enter') select(); }; $('raw-body').append(tr);
      }
      $('table-empty').hidden = records.length > 0;
    }
    function renderPredictionTable() {
      $('prediction-body').replaceChildren(); if (!activeSample) return;
      const top = topMode(); $('prediction-caption').textContent = `目标车辆 ${activeSample.track_id} · 概率 top-1 分支 ${top + 1} · 每步误差单位米`;
      activeSample.prediction_xy.forEach((points, k) => {
        if ($('prediction-choice').value === 'top1' && k !== top) return;
        points.forEach((xy, j) => {
          const truth = activeSample.truth_xy[j], error = Math.hypot(xy[0] - truth[0], xy[1] - truth[1]); const tr = document.createElement('tr');
          for (const v of [activeSample.future_frames[j], k + 1, activeSample.probabilities[k].toFixed(4), ...xy.map(nfmt), ...truth.map(nfmt), nfmt(error)]) { const td = document.createElement('td'); td.textContent = v; tr.append(td); }
          tr.onclick = () => { stop(); frameIndex = frames.indexOf(activeSample.future_frames[j]); render(); }; $('prediction-body').append(tr);
        });
      });
    }
    function drawLoss() {
      if ($('loss-card').hidden) return;
      const loss = $('loss-canvas'), c = loss.getContext('2d'), r = loss.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      loss.width = r.width * dpr; loss.height = r.height * dpr; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, r.width, r.height);
      const rows = meta.reviews[mode].training_history, max = Math.max(...rows.flatMap(v => [v.train_loss, v.val_loss])) * 1.1;
      const px = i => 48 + i * (r.width - 80) / Math.max(rows.length - 1, 1), py = v => 100 - v / max * 75;
      c.font = '11px system-ui'; c.fillStyle = '#607685';
      for (const value of [0,max/2,max]) { c.fillStyle='#607685';c.fillText(value.toFixed(2),8,py(value)+4);c.strokeStyle='#e5ecea';c.lineWidth=1;c.beginPath();c.moveTo(48,py(value));c.lineTo(r.width-32,py(value));c.stroke(); }
      c.fillText('loss', 10, 12); c.fillText('epoch 1', 48, 123); c.fillText(`epoch ${rows.at(-1).epoch}`, r.width - 90, 123);
      for (const [key, color, label] of [['train_loss', '#1764c0', '训练'], ['val_loss', '#c76b08', '验证']]) { c.beginPath(); c.strokeStyle = color; c.lineWidth = 2; rows.forEach((row, i) => i ? c.lineTo(px(i), py(row[key])) : c.moveTo(px(i), py(row[key]))); c.stroke(); c.fillStyle = color; c.fillText(`${label} ${nfmt(rows.at(-1)[key])}`, key === 'train_loss' ? 100 : 230, 18); }
    }
    function download(blob, name) { const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
    $('download-all').onclick = () => download(new Blob([bytes], { type: 'text/csv;charset=utf-8' }), meta.filename);
    $('download-case').onclick = () => { const escape = v => /[",\r\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; const text = [headers, ...scene.map(r => r.values)].map(row => row.map(escape).join(',')).join('\r\n') + '\r\n'; download(new Blob([text], { type: 'text/csv;charset=utf-8' }), `INTERACTION_case_${$('case-select').value}.csv`); };
    $('view-mode').onchange = e => setMode(e.target.value);
    $('case-select').onchange = e => selectCase(e.target.value);
    $('sample-select').onchange = e => { const s = meta.reviews[mode].samples.find(s => s.sample_index === Number(e.target.value)); selectCase(s.case_id, s); };
    $('prediction-choice').onchange = () => { draw(); renderPredictionTable(); };
    $('random-case').onclick = () => { const choices = availableIds.filter(id => id !== $('case-select').value); if (choices.length) selectCase(choices[Math.floor(random() * choices.length)]); };
    $('seed').onchange = () => { rngState = Number($('seed').value) >>> 0; };
    $('track-select').onchange = e => { selectedTrack = e.target.value; render(); };
    $('frame-slider').oninput = e => { stop(); frameIndex = Number(e.target.value); render(); };
    $('play').onclick = play; $('speed').onchange = () => { if (timer !== null) { stop(); play(); } };
    for (const id of ['show-others', 'show-labels', 'show-map', 'show-errors']) $(id).onchange = draw;
    $('table-mode').onchange = renderTable; $('zoom-in').onclick = () => zoom(.75); $('zoom-out').onclick = () => zoom(1 / .75); $('reset-view').onclick = () => { resetBounds(); draw(); };
    function nearest(e) { const box = canvas.getBoundingClientRect(), x = e.clientX - box.left, y = e.clientY - box.top; return hitTargets.map(t => ({ ...t, d: Math.hypot(t.x - x, t.y - y) })).filter(t => t.d < 18).sort((a, b) => a.d - b.d)[0]; }
    canvas.onclick = e => { const t = nearest(e); if (t && mode === 'raw') { selectedTrack = t.r.track_id; $('track-select').value = selectedTrack; render(); } };
    canvas.onmousemove = e => { const t = nearest(e), tip = $('tooltip'); canvas.style.cursor = t ? 'pointer' : 'default'; if (!t) { tip.style.display = 'none'; return; } tip.style.display = 'block'; tip.style.left = Math.max(0, Math.min(t.x + 12, canvas.clientWidth - 180)) + 'px'; tip.style.top = Math.max(5, t.y - 70) + 'px'; tip.textContent = `车辆 ${t.r.track_id} · 帧 ${t.r.frame_id}\nx ${t.r.raw.x} m · y ${t.r.raw.y} m\n速度 ${Math.hypot(t.r.vx, t.r.vy).toFixed(2)} m/s`; };
    canvas.onmouseleave = () => { $('tooltip').style.display = 'none'; };
    canvas.onkeydown = e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); stop(); frameIndex = Math.max(0, Math.min(frames.length - 1, frameIndex + (e.key === 'ArrowRight' ? 1 : -1))); render(); } else if (e.code === 'Space') { e.preventDefault(); play(); } };
    new ResizeObserver(() => { if (scene) { draw(); drawLoss(); } }).observe($('chart-wrap'));
    window.addEventListener('pagehide', stop);
    setMode(meta.default_mode || 'raw'); document.documentElement.dataset.ready = 'true';
  } catch (error) {
    const p = document.createElement('div'); p.className = 'error'; p.setAttribute('role', 'alert'); p.textContent = '无法载入地图或轨迹：' + error.message; $('app').replaceChildren(p); console.error(error);
  }
})();
