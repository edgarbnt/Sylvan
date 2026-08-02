# ÉTAT DES LIEUX — handoff courant (2026-08-02)

> Constat au PRÉSENT, réécrit et non empilé. L'historique long vit dans `memory/` ; l'état des
> modules dans `tools/archi_hud/architecture.json` (source de vérité — la carte porte l'ÉTAT, les
> design-docs portent le POURQUOI). Ce fichier dit **où on en est** et **quoi faire ensuite**.

## Mission
ALife émergente dans un world-model JEPA : l'entité décide elle-même (faim → chercher → approcher →
survivre) par planification dans un WM appris. La locomotion est un prérequis DONNÉ.

## À lire d'abord
1. `tools/archi_hud/architecture.json`, module `world_model` §limites — **la dette n°1**.
2. `diagnostics/diag_viabilite_monde.py` — à passer AVANT tout jugement comportemental.
3. `diagnostics/diag_portee_g0.py` — l'ablation qui désigne le facteur limitant.

---

## 1. LA LIMITE QUI DOMINE TOUT — le WM est aveugle au mouvement des objets
`command_wm.transport_slot` déplace la cible par la **seule ego-motion de l'agent**. Le déplacement
propre d'un objet n'est modélisé nulle part ⇒ **dans le rêve, une proie qui fuit à 0,023 m/pas est
immobile et attend**. Arriver lentement y paraît donc équivalent à arriver vite, et le terme
d'énergie du score tranche l'égalité en faveur de la commande LENTE : la vitesse tombe à
0,025 m/pas dans le dernier mètre — **exactement la vitesse de la proie** — et le rapprochement net
devient NUL. Elle orbite.

**Ablation factorielle** (`diag_portee_g0.py`, dans le budget métabolique réel de 390 pas) :

| facteur | coût en capture |
|---|---|
| **ralenti terminal** | **−64,5 pts** |
| rayon de braquage (2,00 m pour une bouche de 1,00 m) | −23,3 pts |
| erreur de visée (23°) | −2,0 pts |
| intermittence de la vue (50 %) | −0,0 pt |

⇒ **La perception vaut 2 points, la mémoire zéro.** C'est ce qui explique rétroactivement que trois
chantiers (perception apprise, arbitrage, mémoire spatiale) n'aient rien pu démontrer : tous
butaient sur ce plafond.

**Échafaudage posé** — `SYLVAN_PLANNER_SPRINT`, défaut OFF, bannière obligatoire : plancher de
vitesse vx≥0,6 sous 2 m, soit **sa vitesse d'approche existante, aucune capacité nouvelle**.
A/B 36 vies/bras (contrôle d'action passé : 0,250 → 0,600) : survie 828→1358, repas 68→117,
boissons 47→93, vies pleines 6→12, conso par 1000 pas VÉCUS 3,23→3,60. Deux mesures à p<0,05
nominal, **aucune ne survit à Bonferroni** — effet cohérent en signe partout, mais **pas établi**.
🚨 **C'est une DETTE, pas un acquis** : il masque la cécité, il ne la corrige pas.

**→ CHANTIER QUI LA DISSOUT** : un transport de slot conscient du mouvement propre de l'objet.
**Auto-supervisé et JEPA-pur** — le déplacement d'une proie est observable entre deux rétines
consécutives, zéro label requis.

---

## 2. Ce que l'entité EST aujourd'hui (mesuré)
- Corps cinématique (vx, ω) ; monde `foret_v1`, cône 120°, 4 teintes de **proies mobiles**.
- Survie médiane ~350/3000 sans échafaudage, ~430 avec. Distribution **BIMODALE** : 29/36 vies
  meurent avant 600 pas, 6/36 vont au bout, presque rien entre. ⚠️ **La survie est donc un mauvais
  thermomètre** — juger sur des métriques PAR TEMPS VÉCU.
- Perception : slot à 1,42 m / **23,1°** de gisement ; un rayon touche réellement la cible dans
  40 % des ticks seulement.

## 3. Chantiers du 2026-08-02
| chantier | verdict |
|---|---|
| **Monde invivable** | 🛠️ **RÉPARÉ** — `restore_per_item` n'était exporté qu'à la NOURRITURE : l'eau restait à 40 quand la nourriture était passée à 140. Il fallait 50 m de trajet par 1000 pas pour 47 m parcourables ⇒ **aucune entité ne pouvait survivre**. Après : 27 m sur 47 (58 %). **3ᵉ panne silencieuse de la même famille** (l'eau n'hérite pas d'un réglage donné à la nourriture). |
| **Perception de la faim apprise** | ✅ **PARITÉ + MODULARITÉ** — AUC 0,893 sur la seule consommation vécue, ρ̂=0,97 m **découvert** (= eat_radius, jamais dit). Gisement 23,0° contre 23,2° codé-main. **G-mod** : monde repeint → l'appris tient à 24,0°, la règle codée-main s'effondre à 33,3° et perd **91 %** de ses cibles. Promouvable ; reste OPT-IN (`SYLVAN_SLOT_DRIVE_SALIENCY`). |
| **Arbitrage homéostatique** | 🛑 **NÉGATIF INFORMATIF** — G0 : la règle change **51 %** des décisions. G2 : le taux d'acquisition ne bouge pas (3,23→3,30, p=0,877). Changer la moitié des choix sans bouger l'acquisition **prouve que le choix n'est pas le goulot**. Acquis : « socle designé + hystérésis CONSERVÉE » **n'exporte pas** son échec, contrairement au remplacement de juillet. |
| **Portée / approche** | ⭐ voir §1 — le seul levier qui bouge le forage. |

## 4. Négatifs bankés — NE PAS refaire
- **Interception** (viser où la proie SERA) : **+0,0 %** de capture à tous les ratios.
- **Norme homéostatique n/m** pour arbitrer 2 pulsions symétriques : **inerte, démontré** — `D` est
  Schur-convexe ⇒ se réduit toujours à « la jauge la plus démunie d'abord », quels que soient n, m, γ.
- **Tête apprise REMPLAÇANT l'ordre de cible** (juillet) : critère visé atteint, échec **EXPORTÉ**
  (danger 5→13, conso 108→96).
- **Agilité du corps** ×2 : rien ; ×4 : **dégrade** (10 repas contre 22).
- **replan-every 60** : ne transfère pas ici (p=0,351).
- **`wm_remeasured.pt`** est PIRE que `wm_best.pt` (2,12 m / 27,3° contre 1,42 / 23,1).

## 5. Règles de mesure acquises aujourd'hui (chèrement)
1. **Vérifier qu'un bras expérimental a AGI avant de lire son verdict.** Le 1ᵉʳ A/B sprint jugeait
   un bras inerte — posé sur une sortie du planner atteinte **16,8 %** du temps. Une bannière prouve
   le CHARGEMENT, pas l'EXÉCUTION.
2. **Une table plate = un test qui ne mesure rien.** A démasqué deux diagnostics vides (norme
   homéostatique inerte ; capture à 100 % faute de budget réaliste).
3. **Passer `diag_viabilite_monde.py` AVANT tout jugement comportemental** — un monde invivable
   avait **INVERSÉ** le verdict de l'A/B perception (appris perdant 482 vs 724, gagnant après fix).
4. **Fixer la STATISTIQUE avec le seuil** (médiane par vie ≠ poolé) et **donner une MAGNITUDE aux
   KILL**, pas une direction (le KILL sprint s'est déclenché sur 1→2 morts, du bruit).
5. **Le simulateur n'est pas déterministe** : deux runs identiques divergent ⇒ jamais de comparaison
   de trajectoires appariées, toujours graines + vies.
6. **Ne pas étendre un échantillon après avoir vu la tendance** (p-hacking) : pré-déclarer.
7. **Ne pas juger sa propre étiquette** : la vérité per-rayon tirée de `food_rel0` est faussée dans
   les DEUX sens (autres fruits comptés négatifs 21,7 % ; occulteurs comptés positifs 22,6 %).

## 6. Prochain pas — cheaper-first
**Le mouvement des objets dans le WM** (dissout la dette n°1, §1). **Test gratuit d'abord** : le
déplacement propre d'une proie est-il prédictible depuis deux rétines consécutives, et à quelle
précision ? Si oui → tête auto-supervisée sur le slot, **WM gelé**. Si non → dire pourquoi AVANT de
coder quoi que ce soit.

⚠️ Second facteur mesuré et non traité : le **rayon de braquage** (−23,3 pts) — elle ne peut pas
virer à l'intérieur de sa propre allonge sans ralentir. À reprendre APRÈS, séparément.

## Critère de succès (le BUT)
Que l'entité **vive de son forage** dans `foret_v1` **sans échafaudage** : survie non bimodale,
consommation par temps vécu ≥ celle obtenue avec l'échafaudage (3,60 / 1000 pas), et
`SYLVAN_PLANNER_SPRINT` **retirable sans perte**.
