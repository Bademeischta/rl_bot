# Collision-Meshes

RocketSim braucht die Arena-Kollisionsmeshes (`collision_meshes/soccar/mesh_0..15.cmf`).

**Phase 0:** Verwendet werden die Meshes, die `rlgym-rocket-league` mitliefert
(`.venv/Lib/site-packages/rlgym/rocket_league/sim/collision_meshes`, 16 Dateien), kopiert nach `C:\RLbot\collision_meshes`.

**Nach einem Rocket-League-Patch, der die Arena ändert** (spätestens bei der UE6-Migration 2027), müssen die Meshes neu aus der eigenen Installation gezogen werden:

1. RLArenaCollisionDumper holen: https://github.com/ZealanL/RLArenaCollisionDumper
2. Rocket League **offline ohne EAC** starten (Epic-Launcher → „Launch without EAC"), in ein Freeplay-Match auf DFH Stadium gehen.
3. Den Dumper ausführen. Er schreibt `collision_meshes/soccar/*.cmf`.
4. Nach `C:\RLbot\collision_meshes` kopieren und `pytest tests` laufen lassen.
5. Wenn sich die Hashes geändert haben, alte Checkpoints kritisch prüfen: Die Physik ist dann nicht mehr identisch zum Training.

Nie mit aktivem EAC oder online. Das ist ein Ban-Risiko.
