# PROMPT — session « perception pure de la faim »

*Copier le bloc ci-dessous tel quel dans une session neuve.*

---

Salut. Chantier : **dissoudre la dernière clé-apparence codée-main du projet** —
`color_queries = [[1,0,0],[0,0,1],[0,1,0]]` dans `python/sylvan/models/slot_head.py`, qui dit
en dur « rouge = bouffe, bleu = eau, vert = danger ».

**Lis d'abord `docs/design_perception_pure_faim.md`** — tout y est : le but, l'état mesuré, ce
qui est déjà réfuté, la recette et les gates pré-enregistrés. Puis `CLAUDE.md`.

## Ce que je veux, et ce que je ne veux pas

C'est un chantier de **MODULARITÉ, pas de performance**. Le test qui compte :

> j'ajoute un fruit bleu → l'entité le mange → elle apprend qu'il nourrit → **zéro ligne de code touchée**

Je ne veux pas de règle-couleur écrite à la main, ni de gate à la main. Ce qui peut être appris
et qui a un sens à être appris doit l'être. J'accepte un composant appris un peu moins performant
que son équivalent designé (doctrine du 2026-07-22) ; je refuse un composant nuisible.

## La recette — elle a DÉJÀ marché ici

`python/scripts/train_danger_saliency.py` est le seul module de perception marqué `pur` dans la
carte d'archi, jugé **PASS 41/9**, et il a dissous la règle « danger = vert ». Sa forme :

```
P(dégâts | rétine) = σ( b + max_{rayons touchants k}  s(rgb_k) · g(d_k) )
```

`s(rgb)` = MLP 3→16→1 sur **la couleur seule** (donc ce qu'elle apprend de près vaut à toute
distance) · `g(d)` = portée apprise · **max-pooling** (la somme a été testée et a échoué) ·
prior de parcimonie. C'est du Multiple Instance Learning (Ilse et al. 2018).

**Fais la même chose pour la faim**, étiquette = la consommation vécue (`wm.ate`) :

```
P(je consomme bientôt | rétine) = σ( b + max_k  s_faim(rgb_k) · g_faim(d_k) )
```

Puis dans `slot_head.py`, remplacer `cos(rgb, requête) > query_thr` par `s_faim(rgb) > 0.5`.
**La géométrie soft-argmax ne bouge pas** — mesurée excellente (0,24 m / 5,0° avec une sélection
parfaite). Une tête par pulsion : faim, soif, danger.

## Étapes

1. `python/sylvan/models/drive_saliency.py` — jumeau de `DangerSaliency`, générique par pulsion
2. `python/scripts/train_drive_saliency.py` — jumeau du trainer, étiquette = consommation
3. Lancer les gates du §6 du design (G-cons, G-loc, G-gis, **G-mod**)
4. Si tout passe : brancher dans `slot_head.py`, puis A/B closed-loop
5. Commiter le verdict — **négatif compris**, avec ses chiffres

## Garde-fous qui ont déjà coûté cher aujourd'hui

- **Split par ÉPISODE, jamais aléatoire.** À 0,05 m/tick les ticks voisins sont quasi identiques :
  une sonde a rendu 0,42 m en split aléatoire contre 2,58 m par épisode. Facteur 6, verdict inversé.
- **Ne jamais recoder la géométrie du transport** — appeler `wm.transport_slot`. Un diagnostic qui
  la recodait a rendu un faux KILL (5e convention sur 8).
- **Utiliser TOUT le corpus** : `foret_v1{,b,c}_*` = 271 731 ticks / 534 repas. Pas seulement
  `gate_foret_cl` (9 282 ticks) — c'est l'erreur de la session précédente.
- `food_rel0` et la règle-couleur sont des **oracles d'ÉVAL uniquement**, jamais d'entraînement.
- WM **gelé** (`wm_foret_v2_slot`) : on n'entraîne que la tête.
- `bash scripts/... ` : lancer un entraînement en background = la commande python SEULE.

## L'outil de bilan

`PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python diagnostics/diag_bilan.py`

Quatre sections : substrat / perception / vie / pureté. Il applique les calages du slot et découpe
par épisode. C'est lui qui dit où on en est avant et après.

## Le risque, dis-le-moi si tu le rencontres

Le danger disposait de **9 372** ticks de dégâts ; la faim n'a que **534** repas (0,20 % des ticks).
17× moins de signal. Si G-cons échoue, ne conclus pas « l'appris ne marche pas » — conclus que ce
monde ne rend pas assez de conséquence pour qu'on apprenne à voir, et dis-moi quel réglage de MONDE
le corrigerait. Un négatif chiffré et commité vaut mieux qu'un tweak enchaîné.

Prends le temps de mesurer avant de lancer quoi que ce soit de long.
