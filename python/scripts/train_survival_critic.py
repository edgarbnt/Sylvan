"""Entraîne le CRITIQUE DE SURVIE drive-symétrique (chantier critique-appris, 2026-07-05).

Le « trainable critic » de LeCun pour Mode-2 : apprend du VÉCU « combien d'avenir depuis cet
état » — destiné à remplacer la queue analytique (alternance+drain codées-main) du coût survie.
Drive-SYMÉTRIQUE : un token par pulsion [niveau, dist, cos(brg), sin(brg), connu] → encodeur
partagé + pooling → une 3ᵉ pulsion = un token de plus, zéro retrain (contrat Gate-S).

Données = le vécu déjà loggé (buffers hesit_probe_* : plan.food/water crus + drives par replan).
Labels = pipeline G2/B0 : G = 1−γ^(replans restants) + surv (mort ≤10 replans). AUCUN oracle :
positions = ce que le planner CROYAIT (slots), issues = ce qui est réellement arrivé.

Gates offline pré-enregistrés (avant run, principe de travail n°1) :
  1. AUC(V, surv) held-out ≥ 0.85 (référence G2 : 0.88)
  2. NON-SATURATION (le décisif) : sur les replans où les 2 ordres analytiques étaient SATURÉS
     (sf,sw ≥ 2999, écarts 1-6 pts = la racine de l'errance-du-repu), le critique garde ≥ 50%
     de son pouvoir discriminant global (std_V_saturés / std_V_tous ≥ 0.5).
  3. Équilibre drives : ΔV négatif pour CHAQUE drive bas (contrefactuel sur tokens).
  4. Arbitrage swap (B0) : V préfère la config où la ressource de la pulsion basse est proche ;
     rappel B0 : la valeur statique seule avait ÉCHOUÉ ce test (hasard) — ici les features sont
     les tokens du critique, le verdict dira si la forme drive-symétrique fait mieux ;
     le juge FINAL reste la Phase B (critique au bout du rollout, closed-loop).

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_survival_critic \
            [--glob 'data/replay_buffer/hesit_probe_*_surv'] [--out data/checkpoints/survival_critic]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import os
from pathlib import Path

import torch
from torch import nn

GAMMA = 0.99            # par replan (10 pas Godot) — même échelle que G2/B0
H_SURV = 10             # « vivant dans 10 replans » (= 100 pas Godot, G2)
LOW, HIGH = 0.30, 0.50
TOK_DIM = 5             # [niveau, dist/10, cos, sin, connu]
STEPS_PER_REPLAN = 10   # 1 replan planner = 10 pas Godot (cf H_SURV)


def token(level: float, pos: list[float] | None) -> list[float]:
    """Token drive-symétrique : [niveau, dist/10, |sin(bearing)|, cos(bearing), connu].

    SYMÉTRIE MIROIR IMPOSÉE PAR CONSTRUCTION (2026-07-08) : on donne |sin| et non sin.
    Gauche/droite est une SYMÉTRIE EXACTE du monde : la VALEUR d'un état (« combien d'avenir
    depuis ici ») ne peut pas dépendre du côté où se trouve la ressource — seule l'ACTION le peut.
    Avec sin SIGNÉ, le réseau POUVAIT distinguer les deux, et il l'a fait : mesuré sur le critique
    précédent, écart miroir jusqu'à 0.13 (V=0.825 bouffe à droite vs 0.703 à gauche, situations
    physiquement IDENTIQUES) — soit PLUS que l'effet de distance sur 3 m. Ce bruit appris créait un
    optimum de valeur HORS-AXE (~30°) : le planner était récompensé de garder la ressource de biais
    → ORBITE au lieu de foncer, et l'agent ratait le dernier mètre. Symétriser le critique existant
    a suffi à ramener l'optimum PILE DEVANT (0°) et à rendre V monotone en |bearing| à 2 m et 4 m.
    ⚠️ Ce token est construit à DEUX endroits qui DOIVENT rester identiques : ici (entraînement) et
    dans command_planner.py (inférence, branche critic_mode). Toute divergence = train ≠ déploiement.
    (Leçon récurrente du projet : une symétrie connue s'IMPOSE, elle ne se fitte pas — cf slot_calib
    et le readout géométrique.)
    """
    if pos is None:
        return [level, 1.0, 0.0, 0.0, 0.0]
    d = math.hypot(pos[0], pos[1])
    return [level, min(d, 10.0) / 10.0, abs(pos[0]) / (d + 1e-6), pos[1] / (d + 1e-6), 1.0]


def load(dirs: list[str]) -> tuple[torch.Tensor, ...]:
    """→ (X [N,2,TOK], G [N], S [N], EID [N]) ; épisodes coupés aux respawns (drives ↑ ensemble)."""
    X, G, S, EID = [], [], [], []
    eid = 0
    for d in dirs:
        f = Path(d) / "ep_0000.jsonl"
        if not f.exists():
            continue
        rows = []
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("plan")
            if p is None:
                continue
            rows.append((float(r["obs"]["energy"]) / 100.0, float(r["obs"]["thirst"]) / 100.0,
                         p.get("food"), p.get("water"), p.get("sf"), p.get("sw")))
        # split épisodes par respawn
        segs, cur = [], []
        for i, row in enumerate(rows):
            if cur and (row[0] - cur[-1][0] > 0.5 or row[1] - cur[-1][1] > 0.5):
                segs.append(cur); cur = []
            cur.append(row)
        if cur:
            segs.append(cur)
        for seg in segs:
            L = len(seg)
            if L < 15:
                continue
            death = min(seg[-1][0], seg[-1][1]) < 0.03      # fin par mort (drive ~0) vs troncature
            for t, (e, th, fp, wp, sf, sw) in enumerate(seg):
                X.append([token(e, fp), token(th, wp)])
                G.append(1.0 - GAMMA ** (L - t))
                S.append(0.0 if (death and (L - 1 - t) <= H_SURV) else 1.0)
                EID.append(eid)
            eid += 1
    return (torch.tensor(X), torch.tensor(G), torch.tensor(S), torch.tensor(EID))


def analytic_labels(X: torch.Tensor) -> torch.Tensor:
    """DISTILLATION (2026-07-08) : noter chaque état avec le PROFESSEUR ANALYTIQUE au lieu du vécu.

    POURQUOI. Les labels Monte-Carlo (`G` ci-dessus) viennent de ce qui s'est RÉELLEMENT passé
    ensuite : « depuis cet état, j'ai tenu encore L−t replans ». C'est la vérité, mais c'est BRUITÉ —
    l'agent peut être dans une position excellente et mourir 200 pas plus tard pour une raison sans
    rapport ; la réalité étiquette alors cette bonne situation comme mauvaise, et le critique apprend
    une moyenne polluée. Le coût survie ANALYTIQUE, lui, EST déjà une fonction de valeur (« pas-vécus
    simulés », `_survival_extension`) : il donne une note PROPRE, DENSE et COHÉRENTE à chaque état.

    SONDE DE PLAFOND (faite AVANT d'écrire ceci, principe n°1 — gater le cher derrière le pas-cher) :
    cette même valeur analytique, branchée DIRECTEMENT dans la fente du critique (SYLVAN_CRITIC_ORACLE=1)
    et consommée exactement comme lui, forage 33 consommations contre 20 pour le critique appris (et 41
    pour la formule codée-main dans SA propre fente). Donc (a) la fente du critique FONCTIONNE — le
    planner sait exploiter un bon juge ; (b) c'est bien la VALEUR APPRISE le goulot, avec ~65% de forage
    à récupérer. La distillation vise ce plafond de 33.

    HONNÊTETÉ (§2). Distiller un coût codé-main, c'est BLANCHIR de la connaissance codée-main dans des
    poids : on ne peut pas DÉPASSER son professeur (qui plafonne). Ça ne vaut donc PAS comme solution
    finale — c'est un AMORÇAGE et un DIAGNOSTIC. La suite pure = affiner ensuite sur le vécu réel, et
    MESURER que ça dépasse le professeur (sinon on n'a gagné qu'une copie plus lente et plus opaque).

    Les paramètres du monde imaginé sont ceux du DÉPLOIEMENT (drain/restore passés par les scripts) —
    toute divergence recréerait un train ≠ déploiement.
    """
    from sylvan.control.planning.command_planner import CommandPlanConfig, _survival_extension

    cfg = CommandPlanConfig()
    drain = float(os.environ.get("SYLVAN_PLANNER_DRAIN", "0.0005"))
    restore = float(os.environ.get("SYLVAN_PLANNER_RESTORE", "0.4"))

    lvl = X[:, :, 0]                                  # [N, 2] niveaux (énergie, soif)
    dist = X[:, :, 1] * 10.0                          # [N, 2] distances (le token porte dist/10)
    cos_b = X[:, :, 3]                                # cos(bearing) par ressource
    known = X[:, :, 4] > 0.5
    FAR = 1e4                                         # ressource inconnue → hors de portée (= token connu=0)
    df = torch.where(known[:, 0], dist[:, 0], torch.full_like(dist[:, 0], FAR))
    dw = torch.where(known[:, 1], dist[:, 1], torch.full_like(dist[:, 1], FAR))
    # |bearing| depuis cos (le token est mirror-symétrique : le SIGNE est perdu, mais seul |bearing|
    # compte pour le temps de virage — c'est précisément ce que la symétrie miroir garantit).
    bf = torch.acos(cos_b[:, 0].clamp(-1.0, 1.0))
    bw = torch.acos(cos_b[:, 1].clamp(-1.0, 1.0))
    rate = max(cfg.surv_turn_rate, 1e-6)
    # distance bouffe↔eau : inconnue depuis les tokens (le signe du bearing est perdu) → on prend une
    # borne géométrique moyenne. Approximation ASSUMÉE et flaggée : elle n'affecte que la phase-2 du
    # professeur (l'alternance après la 1ʳᵉ ressource), pas le terme dominant (atteindre l'urgente).
    dist_fw = float(torch.sqrt((df.clamp(max=10.0) ** 2 + dw.clamp(max=10.0) ** 2)).mean())
    s_food, s_water = _survival_extension(
        df, dw, lvl[:, 0], lvl[:, 1],
        torch.ones_like(df), torch.zeros_like(df),      # valeur DEPUIS cet état (frais, rien de vécu)
        dist_fw, drain, restore, cfg.nominal_speed,
        cfg.surv_horizon, cfg.surv_margin_weight,
        turn_f=bf / rate, turn_w=bw / rate, gamma=0.0,
    )
    return (torch.maximum(s_food, s_water) / cfg.surv_horizon).clamp(0.0, 2.0)   # même échelle que V


def load_lived(dirs: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """→ (X [N,2,TOK], ACTUAL [N] pas RÉELLEMENT vécus après l'instant, EID [N]).

    Ne garde QUE les vies finies par une MORT. Une vie coupée par la fin du run a une survie
    CENSURÉE (« au moins X ») : la garder étiquetterait une bonne situation comme mauvaise et
    biaiserait le résidu vers le négatif. Même découpe aux respawns que load()."""
    X: list[list[list[float]]] = []
    actual: list[float] = []
    eids: list[int] = []
    eid = 0
    for d in dirs:
        f = Path(d) / "ep_0000.jsonl"
        if not f.exists():
            continue
        rows = []
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("plan")
            if p is None:
                continue
            rows.append((float(r["obs"]["energy"]) / 100.0, float(r["obs"]["thirst"]) / 100.0,
                         p.get("food"), p.get("water")))
        segs: list[list] = []
        cur: list = []
        for row in rows:
            if cur and (row[0] - cur[-1][0] > 0.5 or row[1] - cur[-1][1] > 0.5):
                segs.append(cur)
                cur = []
            cur.append(row)
        if cur:
            segs.append(cur)
        for seg in segs:
            L = len(seg)
            if L < 15 or min(seg[-1][0], seg[-1][1]) >= 0.03:     # trop court, ou CENSURÉ (pas mort)
                continue
            for t, (e, th, fp, wp) in enumerate(seg):
                X.append([token(e, fp), token(th, wp)])
                actual.append((L - t) * STEPS_PER_REPLAN)
                eids.append(eid)
            eid += 1
    return (torch.tensor(X), torch.tensor(actual, dtype=torch.float32), torch.tensor(eids))


def innate_steps(X: torch.Tensor) -> torch.Tensor:
    """La prédiction de l'INNÉ, en pas : « depuis cet état, je vivrai encore N pas »."""
    from sylvan.control.planning.command_planner import CommandPlanConfig

    return analytic_labels(X) * CommandPlanConfig().surv_horizon


def residual_labels(X: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """LABEL RÉSIDU = (survie RÉELLE − prédiction de l'INNÉ), en unités de surv_horizon.

    POURQUOI (2026-07-15, `docs/recherche_critique_argmax.md` §6bis). Le critique ne doit PAS
    ré-apprendre la valeur absolue : 98% en est un socle commun à tous les candidats, qui s'annule
    dans la comparaison — le MSE lui faisait donc optimiser le mauvais 98%. Et il ne peut PAS
    départager les candidats mieux que l'inné, qui est EXACT (écart d'action 1e-5 ≪ erreur d'un
    réseau). Ce qu'il peut apporter, et lui seul, c'est ce que l'inné IGNORE : l'inné suppose un
    trajet DROIT à vitesse nominale avec alternance parfaite, et se croit bon pour 1572 pas quand
    l'entité n'en vit que 930 (optimiste ×1.7). Ce manque-à-vivre est APPRENABLE : R² +0.21 sur des
    vies JAMAIS VUES (diag_experience_residual.py) → ce n'est pas du bruit, c'est une leçon.

    ÉCHELLE : / surv_horizon, pour que le planner puisse faire `note = inné_en_pas + λ·V·surv_horizon`
    (même échelle des deux côtés). Le résidu est NÉGATIF en moyenne (l'inné est optimiste)."""
    from sylvan.control.planning.command_planner import CommandPlanConfig

    return (actual - innate_steps(X)) / CommandPlanConfig().surv_horizon


class SurvivalCritic(nn.Module):
    """Tokens [B, K, TOK_DIM] → V [B]. Drive-symétrique : encodeur PARTAGÉ + somme (invariant
    à l'ordre et au NOMBRE de pulsions → pulsion nouvelle = token nouveau, zéro retrain)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(TOK_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def value(self, toks: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(toks).sum(dim=-2)).squeeze(-1)


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def train_residual(args: argparse.Namespace) -> None:
    """CRITIQUE-CORRECTION : note = coût INNÉ (exact) + λ × correction APPRISE du vécu.

    Le critique n'apprend plus la valeur (impossible : il faudrait trancher un écart de 1e-5), il
    apprend le RÉSIDU — l'erreur systématique de l'inné. Voir residual_labels().

    GATE PRÉ-ENREGISTRÉ (falsifiable, écrit AVANT l'entraînement) : sur des vies JAMAIS VUES,
    `inné + correction` doit prédire la survie réelle MIEUX que `inné` seul, d'au moins +0.10 de R².
    Sinon la correction n'a rien appris d'utile → NE PAS la brancher dans le planner."""
    dirs = sorted(globmod.glob(args.glob))
    X, actual, EID = load_lived(dirs)
    if X.numel() == 0:
        print(f"[critic] AUCUNE vie MORTE (non-censurée) dans {args.glob} — rien à apprendre. STOP.")
        return
    n_ep = int(EID.max()) + 1
    G = residual_labels(X, actual)
    innate = innate_steps(X)
    from sylvan.control.planning.command_planner import CommandPlanConfig
    horizon = CommandPlanConfig().surv_horizon
    print(f"[critic] LABELS = RÉSIDU (vécu − inné). vies={n_ep} instants={len(X)}")
    print(f"[critic] survie réelle méd={float(actual.median()):.0f} pas | inné méd="
          f"{float(innate.median()):.0f} pas → l'inné est optimiste de "
          f"{float(innate.median() / actual.median().clamp(min=1)):.2f}×")

    def fit(tr: torch.Tensor) -> SurvivalCritic:
        torch.manual_seed(args.seed)
        c = SurvivalCritic()
        opt = torch.optim.Adam(c.parameters(), 2e-3, weight_decay=1e-4)
        Xt, Gt = X[tr], G[tr]
        for _ in range(args.iters):
            bi = torch.randint(0, len(Xt), (512,))
            nn.functional.mse_loss(c.value(Xt[bi]), Gt[bi]).backward()
            opt.step()
            opt.zero_grad()
        return c.eval()

    # GATE par VALIDATION CROISÉE 4 PLIS (par ÉPISODE — jamais par instant : les instants d'une même
    # vie sont quasi identiques → un split naïf fuiterait). 57 vies seulement : un pli unique ne teste
    # que ~14 vies, l'estimation serait trop bruitée pour décider. Le CRITÈRE reste +0.10 — on estime
    # mieux la MÊME quantité, on ne déplace pas les poteaux.
    gains, r2i, r2c = [], [], []
    for k in range(4):
        te = (EID % 4 == k)
        c_k = fit(~te)                       # l'entraînement doit rester HORS de no_grad
        with torch.no_grad():
            corr_te = c_k.value(X[te]) * horizon
        a_te, i_te = actual[te], innate[te]
        # RÉFÉRENCE HONNÊTE : on accorde à l'inné SEUL son meilleur recalage affine (son échelle est
        # arbitraire) — sinon on le battrait juste en corrigeant son biais, ce qui serait trivial.
        a_ = torch.stack([i_te, torch.ones_like(i_te)], dim=1)
        coef = torch.linalg.lstsq(a_, a_te.unsqueeze(1)).solution.squeeze(1)
        ri = _r2(i_te * coef[0] + coef[1], a_te)
        rc = _r2(i_te + corr_te, a_te)
        r2i.append(ri)
        r2c.append(rc)
        gains.append(rc - ri)
        print(f"[critic]   pli {k} : inné {ri:+.3f} → inné+correction {rc:+.3f}   gain {rc - ri:+.3f}")

    gain = sum(gains) / len(gains)
    r2_innate, r2_corrected = sum(r2i) / len(r2i), sum(r2c) / len(r2c)
    critic = fit(torch.ones(len(X), dtype=torch.bool))            # modèle final : toutes les vies
    with torch.no_grad():
        corr_te = critic.value(X) * horizon

    print(f"\n[critic] === GATE (moyenne des 4 plis, vies JAMAIS VUES) ===")
    print(f"[critic] R² de l'INNÉ seul (recalé)        : {r2_innate:+.3f}")
    print(f"[critic] R² de INNÉ + CORRECTION           : {r2_corrected:+.3f}")
    print(f"[critic] GAIN                              : {gain:+.3f}  (gate ≥ +0.10)  "
          f"[plis : {', '.join(f'{g:+.3f}' for g in gains)}]")
    print(f"[critic] correction médiane : {float(corr_te.median()):+.0f} pas "
          f"(dispersion {float(corr_te.std()):.0f} pas)")
    if gain < 0.10:
        print("[critic] ❌ GATE ÉCHOUÉ → la correction n'apporte pas assez. NE PAS la brancher "
              "dans le planner (le critère était écrit avant : on ne déplace pas les poteaux).")
    else:
        print("[critic] ✅ GATE PASSÉ → correction utilisable. Reste à vérifier GRATUITEMENT qu'elle "
              "ne DÉTRUIT pas le classement fin de l'inné : diagnostics/diag_residual_lambda.py")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": critic.state_dict(), "labels": "residual", "tok_dim": TOK_DIM,
                "r2_innate": r2_innate, "r2_corrected": r2_corrected, "gain": gain,
                "surv_horizon": horizon, "dirs": dirs, "drive_symmetric": True},
               out / "critic_best.pt")
    print(f"[critic] sauvé → {out / 'critic_best.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/replay_buffer/hesit_probe_*_surv")
    ap.add_argument("--out", default="data/checkpoints/survival_critic")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", choices=("mc", "analytic", "residual"), default="mc",
                    help="mc = retours Monte-Carlo du VÉCU (bruité mais vrai, défaut) ; "
                         "analytic = DISTILLATION depuis le professeur codé-main (propre mais capé "
                         "à sa performance) — cf docstring de analytic_labels() ; "
                         "residual = CORRECTION de l'inné (le seul apport possible d'un critique — "
                         "cf residual_labels() et docs/recherche_critique_argmax.md §6bis)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    if args.labels == "residual":
        train_residual(args)
        return

    dirs = sorted(globmod.glob(args.glob))
    X, G, S, EID = load(dirs)
    n_ep = int(EID.max()) + 1
    te_mask = (EID % 4 == 3)                                # split par ÉPISODE (déterministe, ~25%)
    tr = ~te_mask
    print(f"[critic] dirs={len(dirs)} épisodes={n_ep} replans={len(X)} (train={int(tr.sum())} test={int(te_mask.sum())})")

    if args.labels == "analytic":
        # DISTILLATION : mêmes ÉTATS, notes du PROFESSEUR au lieu de l'issue vécue.
        G = analytic_labels(X)
        print(f"[critic] LABELS = ANALYTIQUE (distillation du coût survie codé-main) : "
              f"cible méd={float(G.median()):.4f} min={float(G.min()):.4f} max={float(G.max()):.4f} "
              f"std={float(G.std()):.4f}")
        if float(G.std()) < 1e-3:
            print("[critic] ⚠️ CIBLE QUASI CONSTANTE → rien à apprendre, la distillation serait vide. STOP.")
            return

    critic = SurvivalCritic()
    opt = torch.optim.Adam(critic.parameters(), 2e-3, weight_decay=1e-4)
    Xt, Gt = X[tr], G[tr]
    for it in range(args.iters):
        bi = torch.randint(0, len(Xt), (512,))
        loss = nn.functional.mse_loss(critic.value(Xt[bi]), Gt[bi])
        loss.backward(); opt.step(); opt.zero_grad()
    critic.eval()

    with torch.no_grad():
        v = critic.value(X[te_mask])
        s = S[te_mask]
        # 0. FIDÉLITÉ À LA CIBLE (le gate DÉCISIF en distillation) : R² sur le held-out. Un critique
        #    qui ne SAIT PAS reproduire son professeur n'a aucune chance de le remplacer — et si R²
        #    est haut mais que le forage reste mauvais, alors le défaut n'est NI les données NI les
        #    notes : c'est l'architecture ou la façon dont le planner l'interroge. Les 2 issues tranchent.
        gt = G[te_mask]
        ss_res = float(((v - gt) ** 2).sum())
        ss_tot = float(((gt - gt.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        # 1. AUC
        o = torch.argsort(v); rk = torch.empty_like(v); rk[o] = torch.arange(1, len(v) + 1, dtype=v.dtype)
        npos, nneg = float(s.sum()), float((1 - s).sum())
        auc = float((rk[s == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")
        # 2. NON-SATURATION : replans held-out où l'analytique était saturée
        sat = torch.tensor([(r[0] >= 2999 and r[1] >= 2999) if r[0] is not None else False
                            for r in _sat_flags(dirs)])[: len(X)]
        sat_te = sat[te_mask.nonzero().squeeze(-1)] if len(sat) == len(X) else None
        ratio = float(v[sat_te].std() / (v.std() + 1e-9)) if sat_te is not None and int(sat_te.sum()) > 30 else float("nan")
        # 3. équilibre drives (contrefactuel niveau bas/haut par token)
        dvs = []
        for k in (0, 1):
            lo, hi = X[te_mask].clone(), X[te_mask].clone()
            lo[:, k, 0] = 0.15; hi[:, k, 0] = 0.85
            dvs.append(float((critic.value(lo) - critic.value(hi)).mean()))
        # 4. arbitrage swap : une pulsion basse → sa ressource proche doit gagner
        e_lvl, t_lvl = X[te_mask][:, 0, 0], X[te_mask][:, 1, 0]
        onelow = ((e_lvl < LOW) & (t_lvl > HIGH)) | ((t_lvl < LOW) & (e_lvl > HIGH))
        both_known = (X[te_mask][:, 0, 4] > 0.5) & (X[te_mask][:, 1, 4] > 0.5)
        gap = (X[te_mask][:, 0, 1] - X[te_mask][:, 1, 1]).abs() > 0.05
        sel = onelow & both_known & gap
        Xs = X[te_mask][sel]
        sw_ = Xs.clone(); sw_[:, 0, 1:] = Xs[:, 1, 1:]; sw_[:, 1, 1:] = Xs[:, 0, 1:]   # swap positions
        v0, v1 = critic.value(Xs), critic.value(sw_)
        dep_is_e = Xs[:, 0, 0] < LOW
        dep_nearer = torch.where(dep_is_e, Xs[:, 0, 1] < Xs[:, 1, 1], Xs[:, 1, 1] < Xs[:, 0, 1])
        correct = torch.where(dep_nearer, v0 > v1, v1 > v0)
        frac = float(correct.float().mean()) if len(Xs) else float("nan")

    print(f"[critic] 0. FIDÉLITÉ à la cible : R² held-out = {r2:.3f}"
          + ("  <- DÉCISIF en distillation : sait-il seulement reproduire son professeur ?"
             if args.labels == "analytic" else "  (vs retours MC)"))
    print(f"[critic] 1. AUC held-out = {auc:.3f} (gate ≥0.85, réf G2 0.88)")
    print(f"[critic] 2. NON-SATURATION : std_V(saturés)/std_V(tous) = {ratio:.2f} (gate ≥0.5)")
    print(f"[critic] 3. équilibre : ΔV(e bas)={dvs[0]:+.3f} ΔV(t bas)={dvs[1]:+.3f} (les 2 <0)")
    print(f"[critic] 4. arbitrage swap = {frac:.2f} sur n={int(sel.sum())} (gate ≥0.7 ; B0 valeur-plate : hasard)")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": critic.state_dict(), "auc": auc, "nonsat_ratio": ratio,
                "dv": dvs, "swap": frac, "gamma": GAMMA, "tok_dim": TOK_DIM,
                "dirs": dirs, "drive_symmetric": True}, out / "critic_best.pt")
    print(f"[critic] sauvé → {out / 'critic_best.pt'}")


def _sat_flags(dirs: list[str]):
    """Re-parcourt les buffers dans le MÊME ordre que load() → (sf, sw) par replan gardé."""
    for d in dirs:
        f = Path(d) / "ep_0000.jsonl"
        if not f.exists():
            continue
        rows = []
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("plan")
            if p is None:
                continue
            rows.append((float(r["obs"]["energy"]) / 100.0, float(r["obs"]["thirst"]) / 100.0,
                         p.get("sf"), p.get("sw")))
        segs, cur = [], []
        for row in rows:
            if cur and (row[0] - cur[-1][0] > 0.5 or row[1] - cur[-1][1] > 0.5):
                segs.append(cur); cur = []
            cur.append(row)
        if cur:
            segs.append(cur)
        for seg in segs:
            if len(seg) < 15:
                continue
            for (_, _, sf, sw) in seg:
                yield (sf, sw)


if __name__ == "__main__":
    main()
