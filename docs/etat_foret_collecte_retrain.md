# ÉTAT — monde-forêt : collecte et retrain faits, gate A1 échoué

**Date** : 2026-07-25 · **Branche** : `feat/perception-consequence` · Rien n'est allé sur `main`.
Session autonome (owner absent). Tout ce qui suit est **mesuré** ; les hypothèses sont étiquetées.

---

## 1. Ce qui est SOLIDE

| brique | preuve |
|---|---|
| monde forêt complet | contrat de monde **12/12 vert** sur le corpus servi |
| déterminisme | G10 PASS ; re-confirmé par accident (deux collectes → **122 215 ticks** au tick près) |
| calibration ancrée sur le réel | budget de trajet **53,9 m/vie** (terrain 0,635 mesuré), pas les 84,9 m supposés |
| corpus | 450 vies / 122 215 ticks, **0 part dégénérée**, 3 conséquences vécues (313 consommations, 3 970 pts de dégâts sur 110 vies) |
| **WM comme modèle de DYNAMIQUE** | open-loop pos **0,020 m** (h=10) / **0,199 m** (h=50) / 0,322 m (h=80) ; yaw 0,5° ; **jalon @50 ATTEINT** |

Checkpoints : `wm_foret_v1` (base, obs 278 / proprio 133) · `wm_foret_v1_slot` (canal-slot greffé).

## 2. Ce qui ÉCHOUE — gate A1

**Le type ne survit pas à l'encodeur** : **29,1 %** de lecture depuis le latent contre **27,3 %** de
majorité — le hasard. Cible : > 70 %. C'est le gate qui justifiait tout le chantier.

Ce que la mesure a **éliminé** :
- le monde et la rétine : la teinte est **100 % séparable dans la rétine** (cos au prototype 1,0000 au p05) ;
- ❌ **l'instabilité du type** — mon premier diagnostic, **RÉFUTÉ** : la repousse est à 2500 ticks, la
  vie la plus longue fait 827 → *aucune* des 450 vies n'atteint une repousse ; le type était déjà
  stable. La re-collecte après « correctif » a rendu le même total au tick près (no-op parfait).

**Hypothèse restante (non testée à fond) : l'absence de PRESSION.** Rien dans l'objectif n'oblige
l'encodeur à garder l'apparence. Le type n'agit sur le monde qu'au contact, via le multiplicateur
nutritif : **313 événements sur 122 215 ticks = 0,26 %**. Prédire son propre latent sans apparence est
une solution parfaite du JEPA ; VICReg empêche l'effondrement sans exiger *cette* information.
C'est le manque nommé au §6ter : *« un chemin d'apprentissage entre la CONSÉQUENCE et la PERCEPTION »*.
Sonde courte (3 époques avec reconstruction rétine) : 29,1 → 29,6 % — trop court pour conclure.

## 3. Gate A2 — structurellement inapplicable en l'état

`build_typed_slots` exige **K=3** groupes de couleur (une teinte par drive). Or §2.5 impose **4
teintes de nourriture** : le clustering trouve **K=5** (3 rouges + bleu + vert). Et le danger noie la
contingence — **9 372** reliefs de dégâts contre **104** d'énergie, parce qu'un repas est instantané
quand les dégâts durent les ~110 ticks passés dans la zone — si bien que *tous* les groupes se lient
à « damage ». Rien n'a été relâché pour le faire passer.

Deux voies possibles, à trancher par l'owner : autoriser **plusieurs clusters par drive**, ou compter
les **événements** de dégât au lieu des ticks.

## 3bis. Verrou A1 — étapes 1 et 2 du chemin owner, FAITES (2026-07-25)

**Étape 1 — le risque « sonder au mauvais endroit » était RÉEL, deux fois.** La sonde lisait
`predicted_latents[:,0]`, qui n'est pas la sortie de l'encodeur mais `to_latent(GRU(encoder(obs)))`
— déjà passé par le goulot récurrent ; et elle étiquetait avec la palette **v7** au lieu de celle
servie. Corrigé : deux sites rendus, palette choisie sur le corpus (foret_v1, cos 1,0000).

| site | LINÉAIRE | MLP | majorité |
|---|---|---|---|
| **ENCODEUR** | 30,4 % | **33,3 %** | 27,3 % |
| LATENT RSSM | 27,2 % | 29,7 % | 27,3 % |

⇒ C'est la **perception** qui jette l'apparence ; le récurrent finit le travail.

**Étape 2 — l'expérience gratuite répond NON.** Tête auxiliaire décodant la teinte depuis la sortie
de l'encodeur (cible dérivée de la rétine, `--w-hue`, tête non sauvée) :

| pression | encodeur (MLP) |
|---|---|
| w_hue = 0 | 33,3 % |
| w_hue = 5 | 35,7 % |
| w_hue = 50 | **37,3 %** |

Une pression **10× plus forte n'achète que +1,6 pt**. Ça sature très loin des 70 %.

**Trois alternatives écartées par la mesure** (pas par l'argument) :
1. le gradient atteint bien l'encodeur (`total.backward()` inclut le terme, tête dans l'optimiseur) ;
2. ce n'est **pas** la SÉLECTION (mon argmin sur les rayons) : la sonde sans sélection
   « le type k est-il PRÉSENT ? » est aussi au hasard (62-64 % pour des bases 59-73 %) ;
3. ce n'est **pas** un effondrement : rang effectif **38,0/128** (encodeur), **45,3/128** (latent) —
   mieux que le 34/128 cité dans les notes du projet.

⇒ **L'encodeur a la capacité, reçoit le gradient, et n'alloue rien à l'apparence.** Le levier n'est
donc pas une tête de décodage. Restent les pistes nommées par l'owner : pression venant de la
**VALEUR/CONSÉQUENCE** (value-aware model learning, représentations reward-predictive), ou un
encodeur à **ATTENTION** — le slot y arrive, lui, parce qu'il lit la rétine par attention
géométrique explicite au lieu d'un MLP dense.

## 4. Ce que je recommande

1. **La tête de décodage est réfutée — ne pas la re-tenter plus fort.** Le sweep 0/5/50 sature.
2. **Piste ATTENTION (la plus étayée par nos propres données).** Le slot lit la même rétine et y
   trouve la couleur sans peine ; un MLP dense n'y arrive pas. Un encodeur à attention sur les rayons
   (au lieu d'un MLP sur 278 dims concaténées) est la différence architecturale exacte entre ce qui
   marche et ce qui échoue chez nous. Testable sans re-collecte.
3. **Piste VALEUR/CONSÉQUENCE** (value-aware model learning) : faire dépendre la perte de la valeur,
   pas seulement de l'état suivant. Plus proche du north-star §6ter, plus coûteux à bâtir.
4. Le reste du monde est prêt : si A1 se règle, la collecte et le retrain se rejouent tels quels.

## 5. Outils ajoutés cette session

`scripts/pad_wm_gaze.py` (élargit un WM 132→133 sans décaler la rétine) ·
`scripts/graft_slot_channel.py` (greffe le canal-slot — géométrie pure lisant la rétine) ·
`scripts/collect_foret_v1.sh` (collecte MIXTE : planner 50 % / babillage 30 % / exploration 20 %) ·
`scripts/collect_foret_all.sh` · `diagnostics/diag_foret_g11_portee.py` (trajet par repas, avec ancre).
Trois outils lisaient le corpus par-épisode de travers (`critic_corpus`, `guards`, `build_typed_slots`) — corrigés.
