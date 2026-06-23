# Carte falsifiable — ce qui manque au WM pour IMAGINER la rotation (2026-06-21)

> Localise, par des sondes **gratuites** (offline, 0 entraînement), le manque de la **clé de voûte** identifiée
> dans `archi_jepa_etat_des_lieux.md` : un World Model qui prédit la perception future sous l'auto-mouvement.
> Outil : `diag_wm_rotation.py`. WM testé : `wm_rich_fidele_sym`.

## Méthode

Sur de vrais segments de **virage** du replay (|ω| moyen > 0.3, 289 segments), on suit le bearing de la bouffe
(cos = « ahead », +1 = devant) le long du segment, et on corrèle au bearing vrai (food_rel0) :
- **REPR** = bearing décodé des latents **teacher-forced** (vraie obs à chaque pas, mode eval = TF complet) →
  teste la **représentation/encodeur** : le vrai latent porte-t-il le bearing qui tourne ?
- **RÊVE** = bearing décodé des latents **open-loop** (rêve depuis la frame 0, vraies commandes) → teste la
  **dynamique** : le rêve transporte-t-il le bearing à travers la rotation imaginée ?
- Décodeur = sonde fraîche entraînée sur latents TF (même espace `to_latent(hidden)` pour les deux).

## Résultats — ⚠️ CORRIGÉS (la 1ʳᵉ mesure avait une FUITE)

**1ʳᵉ passe (`diag_wm_rotation.py`) — FUITE** : sonde entraînée ET évaluée sur les mêmes 12 épisodes → chiffres
GONFLÉS par mémorisation (REPR +0.44/+0.57, RÊVE +0.18/+0.12). **Ne pas s'y fier.**

**Mesure HELD-OUT propre (`diag_test1_readout.py`, split par épisode 9/3, sans fuite)** :

| | REPR (TF, encodeur) | RÊVE (TF-sonde sur latents OL) | TEST1 (sonde entraînée SUR latents OL) |
|---|---|---|---|
| held-out, virages | **+0.14** | **+0.08** | **+0.08** |

- **TEST 1 (la question-pivot) tranche : RETRAIN.** Une sonde entraînée *directement* sur les latents rêvés ne
  récupère PAS le bearing (+0.08) → l'info est **jetée par la dynamique**, pas juste mal lue (≠ readout cheap).
- **Held-out, tout est bas** (REPR +0.14, rêve +0.08) → le WM ne capture pas la perception-sous-rotation de façon
  **généralisable** (encodeur ET dynamique faibles ; le rêve au niveau du hasard). Caveat : 12 épisodes seulement →
  valeurs bruitées, une part = généralisation de la *sonde* (pas que du WM) → plus de données aiderait aussi la mesure.
- Sonde **uncertainty/reveal** : non concluante (pas assez d'événements) — à revisiter avec des reveals dédiés.

**TEST 2 — audit data (`diag_test2_data_audit.py`)** : rotation **abondante** (42–76% des frames) MAIS l'événement
**acquisition** (bouffe balayée derrière→devant) est **rare : ~425 au total** (retina_forage 32, retina_wm_a 198,
retina_eat_a 195). Le WM a vu de la rotation mais peu d'acquisitions → n'a pas appris le transport difficile.

## Conséquence pour le design de la clé de voûte

Levier **dominant = la DYNAMIQUE (rollout)**, pas d'abord l'encodeur : forcer le rêve open-loop à **transporter le
bearing perçu à travers la rotation**. Pistes pures (à départager par d'autres tests gratuits avant tout retrain) :
- **données riches en rotation** (le replay forage a |ω|~0.36 ; un collecteur de virages dédié densifierait le
  signal — NB un babbling aléatoire fait PIRE, cf mémoire) ;
- **perte de fidélité du rollout ciblée perception** (esprit `--w-rollout` déjà validé pour la pose, à étendre au
  contenu perceptuel/bearing à travers le rollout) ;
- **représentation rotation-aware/équivariante** (levier secondaire, l'encodeur à +0.5 a une marge) ;
- **latent stochastique** (pour l'uncertainty des reveals — à mesurer d'abord proprement).

Convergence : ce même rêve faible explique aussi pourquoi le foraging direct est modeste et pourquoi la recherche
latent-pure ne peut pas être dirigée (`diagnostic_engagement_perception.md`). Réparer le rollout = débloquer les deux.
