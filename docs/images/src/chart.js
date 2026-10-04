// Draw a horizontal bar chart into <svg id="chart">. rows: [{label, value, accent, note, pending}]
function barChart({ rows, max, ticks, unit = "", height }) {
  const svg = document.getElementById("chart");
  const W = 1088, left = 230, right = 140, top = 10, rowH = 46, barH = 22;
  const plotW = W - left - right, H = top + rows.length * rowH + 34;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  const x = v => left + plotW * v / max;
  let s = "";
  for (const t of ticks) {
    s += `<line class="c-grid" x1="${x(t)}" y1="${top}" x2="${x(t)}" y2="${top + rows.length * rowH}"/>`;
    s += `<text class="c-axis" x="${x(t)}" y="${top + rows.length * rowH + 24}" text-anchor="middle">${t}${unit}</text>`;
  }
  rows.forEach((r, i) => {
    const y = top + i * rowH + (rowH - barH) / 2, cy = y + barH / 2 + 5;
    s += `<text class="c-label${r.accent ? " strong" : ""}" x="${left - 16}" y="${cy}" text-anchor="end">${r.label}</text>`;
    if (r.pending) {  // no bar: a bar of arbitrary length would read as a score
      s += `<text class="c-note" x="${left + 4}" y="${cy}" style="fill: var(--clay)">${r.note}</text>`;
      return;
    }
    const w = Math.max(3, x(r.value) - left), rr = Math.min(5, w / 2);
    s += `<path d="M${left},${y} H${left + w - rr} Q${left + w},${y} ${left + w},${y + rr} V${y + barH - rr} Q${left + w},${y + barH} ${left + w - rr},${y + barH} H${left} Z" fill="${r.accent ? "var(--clay)" : "#cfccc1"}"/>`;
    s += `<text class="c-value" x="${left + w + 10}" y="${cy}"${r.accent ? ' font-weight="600"' : ""}>${r.value}${unit}</text>`;
    if (r.note) s += `<text class="c-note" x="${left + w + 10 + String(r.value + unit).length * 9 + 10}" y="${cy}">${r.note}</text>`;
  });
  svg.innerHTML = s;
}
