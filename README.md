# Sylvan V1 - Embodied AGI

Ce dépôt contient le pipeline complet **V-M-C** (Visual-Model-Controller) du projet Sylvan, permettant l'émergence de la locomotion sur un Ragdoll physique via Godot et PyTorch.

## Architecture

- **Visual Encoder (V) :** Encodeur proprioceptif et visuel (CNN/ViT ready).
- **World Model (M) :** Mixture of Experts (MoE) asynchrone basé sur un RSSM (Recurrent State Space Model).
- **Controller (C) :** Acteur-Critique latent s'entraînant dans l'imagination du modèle du monde.

## Installation

Utilisez l'environnement optimisé pour PyTorch (compatible AMD/ROCm) :

```bash
# Activation de l'environnement (ex: env_pytorch)
source env_pytorch/bin/activate

# Installation du projet en mode éditable
pip install -e ./python
```

## Utilisation (Cycle Circadien)

L'entraînement automatisé s'effectue via une commande unique qui gère la collecte de données le jour (Godot) et l'apprentissage la nuit (PyTorch).

### Lancer l'entraînement (GPU AMD/ROCm)
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
env_pytorch/bin/run-sylvan --num-cycles 50 --steps-per-day 5000 --epochs-per-night 10
```

### Lancer l'entraînement (CPU Fallback)
```bash
HIP_VISIBLE_DEVICES="" env_pytorch/bin/run-sylvan --num-cycles 50 --steps-per-day 5000 --epochs-per-night 10
```

## Visualisation et Validation

Pour voir l'agent évoluer en temps réel dans l'éditeur Godot :

1. Lancez le serveur de validation :
   ```bash
   ./start_godot_validation.sh
   ```
2. Dans Godot, appuyez sur **F5**. L'agent se connectera au serveur Python pour recevoir ses actions.

## Structure du Projet

- `godot/` : Moteur physique (Jolt), Ragdoll (`sylvan_agent.gd`) et gestion des épisodes.
- `python/sylvan/` :
    - `models/` : RSSM, SparseMoE, Encoders.
    - `control/` : Imagined Rollouts, Actor-Critic, Objectives.
    - `orchestration/` : Cycles Jour/Nuit unifiés.
- `data/` : Replay Buffer (JSONL), Checkpoints (.pt) et Rapports.

## Phase 3.1 - État Actuel

- **Physique :** Ragdoll complet 7 segments, 6 moteurs (HingeJoint3D).
- **Contrôle :** PD Control (Proportional-Derivative) stable.
- **Proprioception :** 51 dimensions (Vecteurs Up/Forward, Velocities).
- **Mémoire :** RSSM avec propagation d'état latent (Alzheimer bug fixé).
