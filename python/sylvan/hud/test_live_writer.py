"""Test autonome du writer live.json. Lancer : PYTHONPATH=python env_pytorch_3.12/bin/python python/sylvan/hud/test_live_writer.py"""
import json
import os
import tempfile

from sylvan.hud.live_writer import write_live


def test_writes_payload_and_is_atomic():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sub", "live.json")  # sous-dossier inexistant → doit être créé
    write_live(path, ts=3, episode=2, step=412, fields={"min_dist": 1.8, "command": [0.7, -0.3]})
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["ts"] == 3, data
    assert data["episode"] == 2 and data["step"] == 412, data
    assert data["fields"]["min_dist"] == 1.8, data
    assert data["fields"]["command"] == [0.7, -0.3], data
    # pas de fichier temporaire laissé derrière
    assert os.listdir(os.path.dirname(path)) == ["live.json"], os.listdir(os.path.dirname(path))


if __name__ == "__main__":
    test_writes_payload_and_is_atomic()
    print("OK: live_writer")
