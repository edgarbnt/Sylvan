# Critères pré-enregistrés — JEPA-ification ÉTAPE 1 (dé-collapse), 2026-06-18

> Écrits AVANT le lancement (discipline anti-boucle, CLAUDE.md §1). Run : `train_wm_jepa.sh`
> → `data/checkpoints/wm_command_hex_v3_jepa`. Mesure : `diag_jepa.py <ckpt>` + `eval_wm_command`.
> Baseline (hex_v2, mesurée au diag gratuit) : eff_rank latent **6.0**/128, cible encodeur **1.45**/128 ;
> displacement RMSE **0.14 cm/pas**, énergie **2.0 %** ; eval open-loop hex_v2 = la référence à ne PAS casser.

## Hypothèse falsifiable
Activer VICReg (var+cov sur les latents RSSM) + perte latente cosine **brise l'effondrement**
(eff_rank ↑) **sans détruire le signal du planner** (displacement/énergie décodés du latent).

## SUCCÈS (→ on passe à l'étape 2 : shift des poids reconstruction→latent, drop radar)
Les DEUX blocs doivent passer :

**A. Dé-collapse réussi**
- eff_rank latent RSSM : **≥ 25 /128** (depuis 6). Idéal ≥ 40.
- eff_rank cible encodeur : **≥ 12 /128** (depuis 1.45) — sinon la cible reste dégénérée et la
  prédiction latente reste triviale même "améliorée".
- lat_std_min > 0 (aucune dimension morte) ; pas de NaN.

**B. Signal planner PRÉSERVÉ (la PORTE — non négociable)**
- `eval_wm_command` : médiane pos **< 0.5 m @50** et **< 1.2 m @100**, yaw médian @100 ≤ ~celui de hex_v2 (+20 % toléré).
- displacement RMSE ≤ **0.20 cm/pas** (depuis 0.14 ; +~40 % toléré), énergie RMSE ≤ **3 %** (depuis 2.0).
- La prédiction latente reste NON triviale : skill ratio (lat_mse / mean-baseline) reste **< 0.3**
  MAIS contre une cible désormais riche (cosine PEUT chuter sous 1.0 — c'est ATTENDU et SAIN :
  prédire une cible non-dégénérée est plus dur ; ce n'est pas un échec).

## KILL / ESCALADE (NE PAS enchaîner un tweak sans nouvelle hypothèse)
- **eff_rank latent reste < 12** après 20 epochs → VICReg trop faible ou effondrement collant.
  Escalade : monter `--vicreg-var/-cov` (2–4), OU `--predictor-arch deep` (asymétrie BYOL), OU gamma ↑.
- **Porte B cassée** (eval pos@50 > 1.0 m, OU displacement RMSE > 0.28, OU énergie RMSE > 4 %) →
  VICReg trop fort / injecte du bruit dans des dims inutiles. Escalade : BAISSER var/cov (0.3–0.5)
  ou gamma (0.5). Le dé-collapse ne doit jamais se payer en précision planner.
- **Divergence** (loss NaN, lat_std explose > ~5) → instabilité → baisser lr ou vicreg.

## Kill PRÉCOCE (gratuit, pendant le run — `grep eff_rank /tmp/wm_jepa.log`)
- Si eff_rank **ne bouge pas de ~6** d'ici l'epoch ~8 → VICReg inopérant → tuer, ne pas finir « au cas où ».
- Si les val losses displacement/energy **doublent** tôt → VICReg trop agressif → tuer, baisser var/cov.

## Après le run
- SUCCÈS A+B → banker `hex_v3_jepa`, passer à l'étape 2 (poids : drop radar 5→0, monter latent,
  garder displacement/énergie en ancres), re-valider planner + nav A→B + arbitrage.
- Tout KILL = un négatif INFORMATIF → STOP + une seule nouvelle hypothèse justifiée par la mesure.

## Ce que ce run NE teste PAS (à garder en tête)
- Il ne déplace PAS encore le travail prédictif vers le latent (poids inchangés) → la "JEPA-ness"
  fonctionnelle (planifier SUR le latent en lâchant les têtes de reconstruction) = étape 2, gatée
  derrière ce succès. Ici on prépare juste le terrain : une représentation non-dégénérée.
