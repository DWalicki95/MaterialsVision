"""
The page a reviewer works through the panels on.

Around two hundred panels have to be looked at, each against eight
criteria, and the verdict that matters is not per panel but per family
and strength setting. Left to a spreadsheet that is two hundred rows to
type by hand, with nothing tying a row to the picture it describes. The
page below removes the typing and keeps the tie: a keystroke records a
panel, the panels roll past in the order they were rendered, and the
verdict on a family is written once its panels have been seen.

**Why a local page and not a hosted one.** The decisive criterion is
whether a wall two pixels across survives being drawn at 0.8 of its
size. That can only be judged at full resolution, so the images must be
shown pixel for pixel - which rules out anything that reduces them to
fit a page, and means the micrographs never have to leave the machine
they were measured on.

**Why decisions are posted to a server rather than kept in the
browser.** Opened as a file, a page has no reliable place to keep
anything: browsers treat each file as its own short-lived origin and
storage can be refused outright. A hundred and eighty verdicts is too
much work to lose that way, so every keystroke is written to disk by
the small server in ``scripts/review_phase0.py``. Browser storage
remains as a fallback, with an export button, for a page opened
without it.

**What a verdict is attached to.** The panel's fingerprint, which is
the hash of the parameters it was rendered with. Change a range and
re-render, and the panels whose numbers changed come back undecided
while the rest keep their verdicts.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# The acceptance criteria of the visual gate, phrased as the failure a
# reviewer would be marking. The order is the plan's; the numbers are
# what the keyboard shortcuts refer to.
CRITERIA = (
    "nie przypomina realnego SEM",
    "nie dalo by sie wiarygodnie zaanotowac",
    "obraz i maska niezgodne",
    "sztuczna krawedz niezgodna z celem",
    "znikaja cienkie sciany albo male pory",
    "fotometria zmienila morfologie",
    "maksymalna sila niewiarygodna",
    "zly wynik po preprocessingu",
)

VERDICTS = ("accepted", "revise", "rejected")

VIEWER_FILENAME = "review.html"


def write_viewer(output_dir: Path) -> Path:
    """Write the review page into a review directory.

    Parameters
    ----------
    output_dir : Path
        Directory holding ``panels.json``.

    Returns
    -------
    Path
        The page.
    """
    path = output_dir / VIEWER_FILENAME
    path.write_text(_page(), encoding="utf-8")
    logger.info("Review page written to %s.", path)
    return path


def _page() -> str:
    """Compose the self-contained page."""
    criteria = "".join(
        f"<label><input type=checkbox data-criterion='{index}'>"
        f"<b>{index + 1}</b> {text}</label>"
        for index, text in enumerate(CRITERIA)
    )
    verdicts = "".join(
        f"<button class=verdict data-verdict='{name}'>{name}</button>"
        for name in VERDICTS
    )
    return _TEMPLATE.replace("__CRITERIA__", criteria).replace(
        "__VERDICTS__", verdicts
    )


_TEMPLATE = """<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>Faza 0 - przeglad plansz</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; font: 13px/1.45 system-ui, sans-serif;
       background: #16181c; color: #e8e8ea; }
#bar { display: flex; gap: 16px; align-items: center; padding: 8px 14px;
       background: #101216; border-bottom: 1px solid #2a2e36;
       position: sticky; top: 0; z-index: 5; }
#bar b { color: #fff; }
#progress { font-variant-numeric: tabular-nums; }
#warn { color: #ffb020; display: none; }
main { display: grid; grid-template-columns: 1fr 280px; gap: 14px;
       padding: 14px; align-items: start; }
#stage img { width: 100%; display: block; border: 1px solid #2a2e36;
             background: #000; }
#pixels { margin-top: 12px; border: 1px solid #2a2e36; overflow: auto;
          max-height: 70vh; background: #000; }
#pixels img { image-rendering: pixelated; display: block;
              width: auto; max-width: none; }
#side { position: sticky; top: 52px; max-height: 88vh; overflow: auto; }
.group { padding: 5px 7px; border-radius: 4px; cursor: pointer;
         display: flex; justify-content: space-between; gap: 8px; }
.group:hover { background: #22262e; }
.group.active { background: #2d3542; }
.group .tag { font-size: 11px; opacity: .75; }
.accepted { color: #6ee787; } .revise { color: #ffb020; }
.rejected { color: #ff6b6b; }
#decide { padding: 10px 14px; background: #101216;
          border-top: 1px solid #2a2e36; position: sticky; bottom: 0; }
#criteria { display: none; grid-template-columns: repeat(2, 1fr);
            gap: 2px 18px; margin: 8px 0; }
#criteria.on { display: grid; }
#criteria label { display: flex; gap: 7px; align-items: center; }
button { font: inherit; padding: 5px 11px; border-radius: 5px;
         border: 1px solid #3a4150; background: #232833; color: inherit;
         cursor: pointer; }
button.on { background: #2f6f3f; border-color: #2f6f3f; }
button.problem.on { background: #8a3a3a; border-color: #8a3a3a; }
input[type=text] { font: inherit; padding: 5px 8px; border-radius: 5px;
                   border: 1px solid #3a4150; background: #1b1f27;
                   color: inherit; width: 100%; }
#meta { font-family: ui-monospace, monospace; font-size: 11.5px;
        white-space: pre-wrap; color: #a9b0bd; margin-top: 8px; }
kbd { background: #2a2f3a; border-radius: 3px; padding: 1px 5px;
      font-size: 11px; }
</style>
</head>
<body>
<div id=bar>
  <b id=title>-</b>
  <span id=progress></span>
  <button id=filter>tylko nierozstrzygniete <kbd>F</kbd></button>
  <button id=zoom>1:1 <kbd>Z</kbd></button>
  <button id=export>eksport JSON</button>
  <span id=warn>brak serwera - decyzje tylko w przegladarce</span>
</div>
<main>
  <div>
    <div id=stage></div>
    <div id=pixels></div>
    <div id=meta></div>
  </div>
  <div id=side></div>
</main>
<div id=decide>
  <div style="display:flex;gap:8px;align-items:center">
    <button id=ok>ok <kbd>A</kbd></button>
    <button id=problem class=problem>problem <kbd>P</kbd></button>
    <span style="flex:1"></span>
    <span id=verdictline>werdykt rodziny:</span>
    __VERDICTS__
  </div>
  <div id=criteria>__CRITERIA__</div>
  <input id=note type=text placeholder="notatka do planszy (Enter zapisuje)">
</div>
<script>
const state = { panels: [], i: 0, decisions: {}, verdicts: {},
                filter: false, zoom: false, server: true };

async function boot() {
  const index = await (await fetch('panels.json')).json();
  state.panels = index.panels;
  try {
    const saved = await (await fetch('review.json')).json();
    state.decisions = saved.decisions || {};
    state.verdicts = saved.verdicts || {};
  } catch (e) { restoreLocal(); }
  render();
}

function restoreLocal() {
  try {
    const raw = localStorage.getItem('phase0');
    if (raw) {
      const saved = JSON.parse(raw);
      state.decisions = saved.decisions || {};
      state.verdicts = saved.verdicts || {};
    }
  } catch (e) { /* storage refused; decisions live in memory only */ }
}

function keep() {
  try {
    localStorage.setItem('phase0', JSON.stringify(
      { decisions: state.decisions, verdicts: state.verdicts }));
  } catch (e) { /* nothing to do; the server copy is the real one */ }
}

async function post(route, payload) {
  keep();
  if (!state.server) return;
  try {
    const response = await fetch('api/' + route, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload) });
    if (!response.ok) throw new Error(response.status);
  } catch (e) {
    state.server = false;
    document.getElementById('warn').style.display = 'inline';
  }
}

function visible() {
  return state.filter
    ? state.panels.filter(p => !state.decisions[p.panel_id])
    : state.panels;
}

function current() {
  const list = visible();
  if (!list.length) return null;
  state.i = Math.max(0, Math.min(state.i, list.length - 1));
  return list[state.i];
}

function decided(panel) {
  const d = state.decisions[panel.panel_id];
  return d && d.fingerprint === panel.fingerprint ? d : null;
}

function render() {
  const panel = current();
  const stage = document.getElementById('stage');
  const pixels = document.getElementById('pixels');
  if (!panel) {
    stage.innerHTML = '<p style="padding:40px">nic do pokazania</p>';
    pixels.innerHTML = ''; renderSide(); return;
  }
  stage.innerHTML = `<img src="${panel.files.figure}">` +
    (panel.files.zoom ? `<img src="${panel.files.zoom}"
       style="margin-top:12px">` : '');
  pixels.innerHTML = `<img src="${panel.files.preview}"
      style="width:${state.zoom ? '1024px' : '100%'}">`;

  const list = visible();
  const done = state.panels.filter(p => decided(p)).length;
  document.getElementById('title').textContent =
    `${panel.family} / ${panel.level} - ${panel.image_id}` +
    (panel.repeat ? ` (powtorzenie ${panel.repeat + 1})` : '');
  document.getElementById('progress').textContent =
    `${state.i + 1}/${list.length} w widoku, ${done}/` +
    `${state.panels.length} ocenionych`;
  document.getElementById('meta').textContent =
    `${panel.note}\\n${panel.material} ${panel.microscope} ` +
    `${panel.scale_bin} | seed ${panel.seed} | fingerprint ` +
    `${panel.fingerprint}${panel.applied ? '' : ' | RODZINA NIE ZADZIALALA'}` +
    `\\nwylosowane: ${JSON.stringify(panel.params)}` +
    `\\nzmierzone: ${JSON.stringify(panel.measurements)}`;

  const decision = decided(panel);
  document.getElementById('ok').classList.toggle(
    'on', !!decision && decision.status === 'ok');
  document.getElementById('problem').classList.toggle(
    'on', !!decision && decision.status === 'problem');
  const criteria = document.getElementById('criteria');
  criteria.classList.toggle(
    'on', !!decision && decision.status === 'problem');
  criteria.querySelectorAll('input').forEach(box => {
    box.checked = !!decision && (decision.criteria || [])
      .includes(Number(box.dataset.criterion));
  });
  document.getElementById('note').value =
    decision && decision.note ? decision.note : '';
  renderSide();
}

function renderSide() {
  const panel = current();
  const groups = {};
  state.panels.forEach(p => {
    const key = p.family + '__' + p.level;
    groups[key] = groups[key] || { total: 0, done: 0, kind: p.kind,
                                   first: p, problems: 0 };
    groups[key].total += 1;
    const d = decided(p);
    if (d) groups[key].done += 1;
    if (d && d.status === 'problem') groups[key].problems += 1;
  });
  const side = document.getElementById('side');
  side.innerHTML = Object.entries(groups).map(([key, g]) => {
    const verdict = state.verdicts[key];
    const mark = verdict ? `<span class="tag ${verdict.status}">` +
      `${verdict.status}</span>` : '';
    const active = panel && key === panel.family + '__' + panel.level
      ? ' active' : '';
    return `<div class="group${active}" data-key="${key}">` +
      `<span>${key.replace('__', ' / ')}` +
      `${g.kind === 'diagnostic' ? ' <span class=tag>diag</span>' : ''}` +
      `</span><span class=tag>${g.done}/${g.total}` +
      `${g.problems ? ' !' + g.problems : ''} ${mark}</span></div>`;
  }).join('');
  side.querySelectorAll('.group').forEach(node => {
    node.onclick = () => {
      const list = visible();
      const at = list.findIndex(
        p => p.family + '__' + p.level === node.dataset.key);
      if (at >= 0) { state.i = at; render(); }
    };
  });
  document.getElementById('verdictline').textContent =
    panel ? `werdykt dla ${panel.family} / ${panel.level}:` : '';
  document.querySelectorAll('.verdict').forEach(button => {
    const key = panel ? panel.family + '__' + panel.level : '';
    const verdict = state.verdicts[key];
    button.classList.toggle(
      'on', !!verdict && verdict.status === button.dataset.verdict);
  });
}

function decide(status) {
  const panel = current();
  if (!panel) return;
  const criteria = [...document.querySelectorAll('#criteria input')]
    .filter(box => box.checked)
    .map(box => Number(box.dataset.criterion));
  const decision = {
    panel_id: panel.panel_id, fingerprint: panel.fingerprint,
    family: panel.family, level: panel.level,
    image_id: panel.image_id, status: status,
    criteria: status === 'problem' ? criteria : [],
    note: document.getElementById('note').value,
  };
  state.decisions[panel.panel_id] = decision;
  post('decision', decision);
  if (status === 'ok') next(); else render();
}

function setVerdict(status) {
  const panel = current();
  if (!panel) return;
  const key = panel.family + '__' + panel.level;
  const verdict = {
    key: key, family: panel.family, level: panel.level,
    fingerprint: panel.fingerprint, status: status,
    reason: prompt('uzasadnienie werdyktu:',
                   (state.verdicts[key] || {}).reason || '') || '',
  };
  state.verdicts[key] = verdict;
  post('verdict', verdict);
  renderSide();
}

function next(step) {
  const list = visible();
  state.i = Math.min(list.length - 1, state.i + (step || 1));
  render();
}

document.getElementById('ok').onclick = () => decide('ok');
document.getElementById('problem').onclick = () => decide('problem');
document.querySelectorAll('.verdict').forEach(button => {
  button.onclick = () => setVerdict(button.dataset.verdict);
});
document.getElementById('filter').onclick = () => {
  state.filter = !state.filter; state.i = 0;
  document.getElementById('filter').classList.toggle('on', state.filter);
  render();
};
document.getElementById('zoom').onclick = () => {
  state.zoom = !state.zoom;
  document.getElementById('zoom').classList.toggle('on', state.zoom);
  render();
};
document.getElementById('export').onclick = () => {
  const blob = new Blob([JSON.stringify(
    { decisions: state.decisions, verdicts: state.verdicts }, null, 2)],
    { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'review.json';
  link.click();
};
document.getElementById('note').onkeydown = event => {
  if (event.key === 'Enter') {
    const panel = current();
    const decision = panel && decided(panel);
    decide(decision ? decision.status : 'ok');
  }
  event.stopPropagation();
};
document.addEventListener('keydown', event => {
  if (event.target.tagName === 'INPUT') return;
  const key = event.key.toLowerCase();
  if (key === 'a') decide('ok');
  else if (key === 'p') decide('problem');
  else if (key === 'f') document.getElementById('filter').click();
  else if (key === 'z') document.getElementById('zoom').click();
  else if (event.key === 'ArrowRight') next(1);
  else if (event.key === 'ArrowLeft') next(-1);
  else if (key >= '1' && key <= '8') {
    const box = document.querySelector(
      `#criteria input[data-criterion="${Number(key) - 1}"]`);
    if (box) { box.checked = !box.checked; decide('problem'); }
  } else return;
  event.preventDefault();
});
boot();
</script>
</body>
</html>
"""
