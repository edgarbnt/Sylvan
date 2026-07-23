GATE DU CRITIQUE SUR LE VECU — resultat 2026-07-23

CE QUI A CHANGE depuis le gate RETRACTE le matin meme :
 1. le VECU reel (40 vies collectees sur le monde gele bosquets_v2, replan=60) au lieu d un
    simulateur maison dont la fidelite n avait jamais ete verifiee ;
 2. la cible compte des EVENEMENTS (repas observes) au lieu d etre derivee d un etat final —
    le selfcheck VERIFIE que manger tot et manger tard comptent pareil (l artefact retracte) ;
 3. un CONTROLE DE COHERENCE BLOQUANT ouvre le gate : « le cout qui fait VIVRE l entite est-il
    correle a ma cible ? ». C est le test qui manquait et qui avait laisse passer un faux positif.

NOTE D INFRASTRUCTURE : en mono-pulsion la branche plan_wm_slot ne loggue QUE target+reason
(verifie) — les coordonnees ne sont PAS dans le corpus. La geometrie est donc lue a la SOURCE,
dans la retine (144 floats/tick), avec les angles du FOV reellement servi (cone 120).

--- RESULTAT (40 vies, 1440 instants de replan, delta=600, split PAR VIE) ---
CONTROLE DE COHERENCE : corr(inne, cible) = +0,502 -> GATE OUVERT (la cible mesure du reel).

  predicteur              R2 (vies jamais vues)
  inne (recalibre)                     +0,234
  tete apprise (ridge)                 +0,327
  gain                                 +0,093   (barre +0,05)

Direction POSITIVE sur 4/4 plis : +0,196 / +0,056 / +0,108 / +0,013.
VERDICT : PASS.

⚠️ TROIS RESERVES, a lire avec le PASS :
 1. **Le gain est FRAGILE** : +0,013 sur le pli 3, soit un quart de la barre. Positif partout,
    mais un pli de plus aurait pu le faire basculer.
 2. **L ABLATION ne trouve AUCUNE feature porteuse** : retirer la faim coute 0,022, le cap 0,016,
    la visibilite 0,029 — et retirer la DISTANCE AMELIORE le score (-0,021). Le gain ne vient donc
    pas d une information identifiable ; c est le signe d un modele qui exploite des combinaisons
    faibles, pas d une comprehension. A surveiller : c est typiquement ce qui ne transfere pas.
 3. **Ce gate ne juge PAS un classement de candidats.** Le vecu n observe qu UN candidat par etat.
    Il dit que la cible VAUT LA PEINE, pas qu une tete saura ranger 117 options — la question qui
    decide vraiment de l utilite d un critique dans un planner MPC.

⇒ Le chantier critique est ROUVERT, sur base saine cette fois. Prochain pas : entrainer une vraie
tete (capacite > ridge) sur cette cible et ce corpus, et surtout construire le juge de CLASSEMENT
qui manque — probablement en instrumentant le serveur pour logger le score par candidat, plutot
qu en fabriquant des contrefactuels (ce qui a deja coute un faux positif).
