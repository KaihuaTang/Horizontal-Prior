// Progressive enhancement only: copy button for the BibTeX block, tooltips for the two SVG charts.
document.querySelectorAll('button[data-copy]').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var el = document.getElementById(btn.dataset.copy), label = btn.textContent;
    if (!el || !navigator.clipboard) return;
    navigator.clipboard.writeText(el.textContent.trim()).then(function () {
      btn.textContent = btn.dataset.done || 'Copied';
      setTimeout(function () { btn.textContent = label; }, 1600);
    });
  });
});

// Tooltip + crosshair for charts produced by scripts/svg_line_chart.py.
// Markup expected: <figure class="chart"> <svg data-chart='…'>…</svg> <div class="tip"></div> </figure>
// with .chart{position:relative} and .tip{position:absolute;visibility:hidden;pointer-events:none}.
document.querySelectorAll('svg[data-chart]').forEach(function (svg) {
  var d = JSON.parse(svg.dataset.chart), host = svg.parentNode, tip = host.querySelector('.tip'), cross = svg.querySelector('.cross');
  if (!tip) return;
  svg.querySelectorAll('.hit').forEach(function (r) {
    function show() {
      var i = +r.dataset.i, cx = +r.getAttribute('x') + +r.getAttribute('width') / 2;
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.style.visibility = 'visible';
      tip.innerHTML = '<b>' + d.x[i] + '</b>' + d.series.slice().sort(function (a, b) { return parseFloat(b.v[i]) - parseFloat(a.v[i]); }).map(function (s) {
        return '<div><i style="display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:' + s.color + '"></i>' + s.name + ' &nbsp;<strong>' + s.v[i] + '</strong></div>'; }).join('');
      var box = svg.getBoundingClientRect(), hb = host.getBoundingClientRect();
      var px = box.left - hb.left + cx / svg.viewBox.baseVal.width * box.width;
      // keep the tooltip off the direct labels at the line ends: flip to the left of the crosshair in the right half
      tip.style.left = (px > box.width * 0.5 ? Math.max(px - tip.offsetWidth - 12, 4) : px + 12) + 'px';
      tip.style.top = (box.top - hb.top + 8) + 'px';
      tip.style.visibility = 'visible';
    }
    function hide() { tip.style.visibility = 'hidden'; cross.style.visibility = 'hidden'; }
    r.addEventListener('mouseenter', show); r.addEventListener('mouseleave', hide);
    r.addEventListener('touchstart', show, { passive: true });
  });
});
