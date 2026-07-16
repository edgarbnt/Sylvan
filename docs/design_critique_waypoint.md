# Design — le critique note les WAYPOINTS (chantier post-G1, 2026-07-16)

## Mission
Remplacer progressivement le scoreur codé-main de l'étage waypoint par une valeur APPRISE du vécu
(forme LeCun `note = inné + λ·correction`). C'est la réhabilitation du critique : ~10 options très
dissemblables (écarts de centaines de pas) au lieu de 33 arcs à 1e-5 — l'étage où un réseau PEUT
discriminer (HIQL fig.8, `docs/recherche_hjepa_waypoint.md` §3).

## À lire d'abord
- `docs/recherche_hjepa_waypoint.md` §6 (« après G1 ») — la licence du chantier.
- `python/scripts/train_survival_critic.py:219` (`residual_labels`) + `:257` (gate CV-4 +0.10) — réutilisés.
- `python/sylvan/control/waypoint_layer.py` — l'étage v2 gaté (G1 PASS 14 repas / 1 mort).

## Fait mesuré qui fonde le design
Corpus déterministe = boucle auto-confirmante (mesuré 2026-07-08 : 18.6 % d'approches alignées →
le critique ne peut pas apprendre ce qu'il ne voit jamais). → L'exploration vit À L'ÉTAGE WAYPOINT
(varier les candidats commités ; leçon Director : le bonus au manager, jamais au worker), collecte
seulement, déploiement déterministe.

## Design
- **Objet noté** : la paire (état, candidat) au moment de la DÉCISION — un Q, pas un V d'état.
- **Label** : pas RÉELLEMENT vécus après la décision (mêmes conventions que `load_lived` : coupe aux
  respawns, vies censurées exclues). Monnaie = pas.
- **Forme IC+TC** : label résiduel = (vécu − `innate_steps(état)`)/H — l'inné d'état est aveugle au
  candidat, donc TOUT l'effet du choix de route (danger traversé, détour payé) vit dans la correction,
  qui seule voit le candidat. Au déploiement : `score(c) = score_analytique_route(c)_en_pas + λ·H·Q(s,c)`
  (le scoreur main reste l'IC de l'étage tant que l'appris n'a pas gagné le droit de le remplacer).
- **Symétrie miroir PAR CONSTRUCTION** (leçon token |sin| : une symétrie connue s'impose, ne se fitte
  pas) : canonicalisation — si wp_x < 0, miroir de TOUTES les x (wp, cible, verts). Géométrie relative
  wp↔cible↔verts préservée, côté aboli.
- **Features candidat (post-canon)** : dist/sin/cos du wp, dist/sin(signé relatif)/cos de la cible,
  longueur totale, **d_vert_leg1, d_vert_leg2 = distance brute du vert perçu le plus proche à chaque
  segment, SANS marge** (les constantes main 1.0/1.4 SORTENT des features — le critique apprend la
  distance létale de ses morts), is_direct. + les 2 tokens drives existants (`token(e,food)`,
  `token(t,water)`).
- **Exploration** : `SYLVAN_WP_EXPLORE_EPS` (défaut 0 = OFF) — par décision, avec prob ε≈0.15, commettre
  un candidat UNIFORME (y compris les mauvais : le critique doit voir « à travers le vert = mort »).
  Un leg dure 100+ ticks → ~1 leg exploratoire sur 5, vies viables, contraste réel.
- **Log décision** : `SYLVAN_WP_LOG=dir` → jsonl par décision {tick, cible, drives, features par
  candidat, coûts analytiques, choisi, explore}. Issue jointe par tick au flux BC.

## ⭐ AMENDEMENT (2026-07-16, post-collecte) — G-GAP ÉCHOUÉ → LE LABEL DEVIENT LA DOULEUR (owner)
Le gate 1 tel qu'enregistré (écart de survie-après > +100 pas) a **ÉCHOUÉ** (gap −345, commit
`f0a6dc6`) : la santé est du SLACK — une traversée coûte 15-30 dégâts, la mort vient de
l'ACCUMULATION → le label survie DILUE structurellement le signal danger (+ biais de phase). La
sonde causale raffinée (Δsanté@200 ticks, jointure logs Godot) montre le signal réel et MONOTONE :
à-travers-vert **28 %** touchés (moy −3.1) / intermédiaire **5 %** / dégagé **0 %**. Arbitrage
owner : **label = DOULEUR (dégâts dans les 200 ticks suivant la décision)**. Gain conceptuel : la
santé gagne enfin un LECTEUR (mesuré : rien ne la lit avant 0), et la « distance létale » codée-main
sort du scoreur — apprise des morsures vécues. Le vert reste dans les FEATURES (percept), sa
LÉTALITÉ devient apprise (flaggé).
- **Entraînement** : Q_douleur(features candidat 10-d) → dégâts@200/100, MSE, mêmes conventions CV.
- **Déploiement (si gates)** : `score(c) = longueur(c) + κ·Q_douleur(c)` — les termes verts main
  (marges 1.0/1.4, W=25) SORTENT du chemin vivant. κ = taux d'échange pas/dégât (ancre : 100 dégâts
  = mort ≈ vie restante ~1400 pas → 14 pas/dégât ; l'aversion au risque multiplie) — **constante
  d'échafaudage flaggée**, défaut κ=100 pas/dégât, jugée par l'A/B.

## Gates PRÉ-ENREGISTRÉS v2 — DOULEUR (écrits AVANT l'entraînement)
1. **G-pain** : AUC(« prendra ≥1 dégât dans les 200 ticks ») > **0.80** sur décisions TENUES
   (CV 4 plis par VIE) ; ET **monotonie** de la douleur prédite sur les buckets de dégagure
   (retrouver 28/5/0 sans les marges main). Échec → ne pas brancher.
2. **A/B closed-loop** (si 1) : scoreur analytique vs longueur+κ·Q_douleur, 12 vies seed 1, monde
   danger : **repas > 10 ET morts ≤ 2 ET ≥ bras analytique (14/1)**. Échec = analytique conservé,
   négatif commité.

## ⭐ VERDICT A/B (2026-07-16 soir) : ÉCHEC — analytique CONSERVÉ, douleur BANKÉE non branchée
Gate 1 (offline) PASSÉ (AUC 0.881, monotone 39.4/3.1/0.0 sans marges main). Gate 2 (closed-loop)
**ÉCHOUÉ** : bras douleur **7 repas / 1 mort / 18 boissons** vs analytique **14 / 1 / 14**. Le
critique égale la main en SÉCURITÉ (9 vies zéro dégât) mais divise le forage par 2 — le troc
évitement↔repas re-rentre. Causes diagnostiquées (pas devinées) : (a) **myopie du label** —
douleur@200 ticks ≈ un leg → un wp qui RETARDE la traversée paraît indolore (vu au smoke K : wp
commité DEVANT le gardien), puis à courte portée tout fait mal → détours dispendieux (aborts 32
vs 14) ; (b) **κ=100 pas/dégât ~3× plus averse** que la main (39 dégâts → 78 m vs ~25 m max) → la
bouffe gardée ne vaut plus le risque → morts de faim 7, re-campement eau (18 boissons, signature
mur-vert@600). PISTES pour la reprise (hypothèses NOUVELLES, pas du tuning) : label à attribution
de route (douleur par leg parcouru, pas fenêtre fixe) ; κ CALIBRÉ des données (pas vécus perdus
par dégât, mesurable du corpus) ; re-A/B seulement après. Checkpoint bankée :
`data/checkpoints/waypoint_pain/pain_best.pt` (opt-in SYLVAN_WP_PAIN_CRITIC, jamais défaut).

## Gates v1 (HISTORIQUE — survie ; le gate 1 a échoué, cf amendement)
1. **G-gap (licence, gratuit post-collecte)** : les issues divergent-elles selon le choix ? Critère :
   écart médian de survie-après-décision entre legs exploratoires et legs argmin, à états comparables,
   **> 100 pas** ; ET morts-danger surreprésentées dans les choix à-travers-vert. Écarts ≈ 0 → **KILL
   du chantier** (rien à apprendre à cet étage non plus), à commiter comme négatif.
2. **G-res** : gate résiduel INCHANGÉ — `inné + correction` > `inné seul` de **+0.10 R²** (CV 4 plis
   par VIE, jamais par instant).
3. **G-rank** : sur paires exploratoires tenues (même état, choix différents), Q ordonne les issues
   avec **AUC > 0.65** (c'est le gate décisionnel : prédire ne suffit pas, cf 2026-07-08).
4. **A/B closed-loop** (SEULEMENT si 2 ET 3) : analytique vs analytique+λQ, 12 vies seed 1, monde
   danger : **repas > 10 ET morts ≤ 2 ET ≥ bras analytique (14/1)**. Échec = on garde l'analytique,
   négatif commité.

## Étapes (le cher gaté derrière le gratuit)
Explo+log (code, smoke offline) → collecte ~24 vies (2 seeds, séquentiel) → G-gap → entraînement
(minutes, CPU) + G-res/G-rank → A/B. Chaque étape s'arrête si son gate échoue.
