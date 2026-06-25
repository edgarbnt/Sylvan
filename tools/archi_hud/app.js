"use strict";

const ETAT_LABEL = { pur: "pur", partiel: "partiel", echafaudage: "échafaudage", manquant: "manquant" };
let MODULES_BY_ID = {};

function nodeEl(m, focus) {
  const el = document.createElement("div");
  el.className = `node etat-${m.etat}` + (m.id === focus ? " focus" : "");
  el.dataset.id = m.id;
  if (m.live_field) el.dataset.liveField = m.live_field;
  const deps = (m.depends_on || [])
    .map((d) => `<span>${(MODULES_BY_ID[d] || {}).titre || d}</span>`)
    .join("");
  el.innerHTML =
    `<div class="titre">${m.titre}</div>` +
    `<div class="quoi">${m.quoi}</div>` +
    (deps ? `<div class="deps">⬆ alimenté par : ${deps}</div>` : "") +
    (m.id === focus ? `<div class="focus-badge">◀ ON BOSSE ICI</div>` : "") +
    `<div class="live-val"></div>`;
  el.addEventListener("click", () => openPanel(m));
  return el;
}

function section(label, html) {
  if (!html) return "";
  return `<div class="sec"><div class="sec-h">${label}</div><div class="sec-b">${html}</div></div>`;
}

function openPanel(m) {
  const body = document.getElementById("panel-body");
  const preuves = (m.preuves && m.preuves.length)
    ? `<ul class="preuves">${m.preuves.map((p) => `<li>${p}</li>`).join("")}</ul>`
    : "";
  const liens = (m.depends_on || [])
    .map((d) => `<span>${(MODULES_BY_ID[d] || {}).titre || d}</span>`)
    .join(" ");
  const limites = m.limites && m.limites !== "—" ? m.limites : "";
  body.innerHTML =
    `<h2>${m.titre}</h2>` +
    `<span class="pill etat-${m.etat}">${ETAT_LABEL[m.etat]}</span>` +
    `<div class="quoi-line">${m.quoi}</div>` +
    section("Rôle", m.role) +
    section("Comment", m.comment) +
    section("Ce qu'il apporte", m.apporte) +
    section("État &amp; pourquoi", m.etat_detail) +
    section("Limites", limites) +
    section("Preuves", preuves) +
    section("Alimenté par", liens) +
    (m.code ? `<div class="code">📄 ${m.code}</div>` : `<div class="code">— pas encore de code —</div>`);
  document.getElementById("panel").classList.remove("hidden");
}

function renderArchitecture(data) {
  MODULES_BY_ID = {};
  data.modules.forEach((m) => (MODULES_BY_ID[m.id] = m));
  const pipeline = document.getElementById("pipeline");
  pipeline.innerHTML = "";
  data.modules.forEach((m, i) => {
    pipeline.appendChild(nodeEl(m, data.focus));
    if (i < data.modules.length - 1) {
      const spine = document.createElement("div");
      spine.className = "spine";
      pipeline.appendChild(spine);
    }
  });
}

function startLivePolling() {
  /* complété en Task 5 */
}

async function init() {
  document.getElementById("panel-close").addEventListener("click", () =>
    document.getElementById("panel").classList.add("hidden")
  );
  const data = await fetch("architecture.json").then((r) => r.json());
  renderArchitecture(data);
  startLivePolling();
}

init();
