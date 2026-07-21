# Prompt de démarrage — session « débloquer et démontrer un choix complexe »

> Copier-coller le bloc ci-dessous comme premier message de la nouvelle session.

---

Salut. On reprend Sylvan. Lis d'abord, dans cet ordre : `ETAT_DES_LIEUX.md` (handoff courant,
2026-07-21), `memory/sylvan-guards.md`, puis `memory/sylvan-foraging-economy.md`.

**Contexte en une phrase** : la session précédente a produit **8 auto-corrections de mesure** — à
chaque fois une constante ou une étiquette CRUE au lieu d'être MESURÉE. Résultat : le chantier
arbitrage a été clos (sa « place » était un artefact), la mémoire a été réhabilitée (×2 en food-only),
et un échafaudage (`far_align`) s'est révélé handicapant en arène ouverte. Des garde-fous
automatiques existent maintenant (`diagnostics/guards.py`) — **utilise-les**.

**But de cette session** : faire franchir à l'entité un cap visible — qu'elle **démontre un choix
complexe** : atteindre une ressource **vue puis cachée par un mur**, via un détour, mieux qu'un agent
sans mémoire. C'est le plus petit exemple concret de « décider », et tout est en place pour le tenter.

**Ordre de marche (cheaper-first, ne pas sauter d'étape) :**

1. **Ligne de base propre** — FA=0 en arène ouverte, `guards.sanity()` + `check_constants()` avant
   tout verdict. Re-mesurer budget/cycle et courbe atteinte-vs-distance. Tous les chiffres de la
   semaine passée sont teintés par `far_align` : il faut une référence saine avant de comparer quoi
   que ce soit.
2. **Consolider la mémoire** (2-3 seeds, multi-drive, monde-mur, FA=1 car il y est PORTEUR) : passer
   le +23 % de « suggestif » à solide. Juge = **courbe d'atteinte** (n en milliers), pas les
   consommations (n en dizaines).
3. **Dégeler le G3 obstacle** avec la mémoire branchée : c'est LA démonstration visée.
4. Si ça passe → promouvoir la mémoire dans la config vivante, et seulement ensuite enrichir le monde.

**Une anomalie gratuite à investiguer en chemin** : en monde-mur, l'entité est **immobile 49 % des
ticks même avec `far_align`**. Elle percute massivement l'obstacle. Personne n'a regardé pourquoi ;
ça peut être un gros gisement.

**Ce sur quoi je veux que tu insistes :**
- **Mesure avant de croire.** Toute constante utilisée dans un jugement doit être vérifiée sur le
  corpus (`check_constants`). Toute anomalie (métrique à 0, entité immobile) = on CREUSE, on ne
  rapporte pas.
- **Juge sur des métriques qui voient** : courbe d'atteinte, ratio d'errance, budget/cycle. La survie
  est un instrument aveugle ici (dérive nulle).
- **Magnitude vs bruit** : pas de « PASS » si l'effet est dans le bruit ; pas de « réfuté » sur un
  effectif de quelques dizaines. Dis « sous-puissant » quand c'est le cas.
- **Pré-inscris** les critères avant de lancer, et ne les déplace pas après.

**Ne refais pas** (négatifs bankés, détail dans `ETAT_DES_LIEUX.md` §7) : rouvrir le critique
d'arbitrage ; espérer que la vitesse règle quelque chose ; toucher au restore ; retirer `far_align`
en monde-mur.

Commence par me proposer ton plan pour l'étape 1, sans rien lancer.
