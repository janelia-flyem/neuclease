"""
Tests for the 'multires-meshes' derived type in neuclease.misc.derived_updates.
"""
from types import SimpleNamespace

import pytest
from confiddler import validate

pytest.importorskip("vol2mesh")

from neuclease.misc import derived_updates  # noqa: E402


def test_config_schema():
    cfg = {'update-derived-types': ['multires-meshes'],
           'multires-meshes': {'lod-error-thresholds': [2, 19], 'source-scale': 1}}
    validate(cfg, derived_updates.ConfigSchema, inject_defaults=True)
    assert cfg['multires-meshes']['lod-error-thresholds'] == [2, 19]
    assert cfg['multires-meshes']['chunk-shape-s0'] == [512, 512, 512]

    # Not updated unless requested
    default_cfg = {}
    validate(default_cfg, derived_updates.ConfigSchema, inject_defaults=True)
    assert 'multires-meshes' not in default_cfg['update-derived-types']


def test_update_multires_body_meshes(monkeypatch):
    calls = []
    affected = SimpleNamespace(changed_bodies=[1, 2], removed_bodies=[3], new_bodies=[4])

    def mutated_bodies(server, uuid, seg, derived_type, ignore_before_uuid=None):
        calls.append(('mutated', derived_type))
        return {'uuid': 'prev', 'mutid': 5}, affected, 9

    monkeypatch.setattr(derived_updates, 'mutated_bodies_since_previous_update', mutated_bodies)
    monkeypatch.setattr(derived_updates, 'init_multires_mesh_instances', lambda *args: calls.append(('init',)))
    monkeypatch.setattr(derived_updates, 'update_multires_body_mesh',
                        lambda server, uuid, seg, body, config, force, processes: calls.append(('update', body, force)))
    monkeypatch.setattr(derived_updates, 'store_update_receipt',
                        lambda server, uuid, seg, derived_type, mutid: calls.append(('receipt', derived_type, mutid)))

    config = {'lod-error-thresholds': [2, 19]}
    derived_updates.update_multires_body_meshes('server', 'uuid', 'segmentation', config, force=True)
    assert calls == [
        ('init',),
        ('mutated', 'multires-meshes'),
        ('update', 1, True), ('update', 2, True), ('update', 3, True), ('update', 4, True),
        ('receipt', 'multires-meshes', 9),
    ]

    # An invalid config is rejected before anything is done.
    calls.clear()
    with pytest.raises(ValueError):
        derived_updates.update_multires_body_meshes('server', 'uuid', 'segmentation', {'lod-error-thresholds': [5, 1]})
    assert calls == []

    # If a body fails, no receipt is stored (so the next update tries again).
    def fail(*args, **kwargs):
        raise RuntimeError("oops")

    monkeypatch.setattr(derived_updates, 'update_multires_body_mesh', fail)
    with pytest.raises(RuntimeError):
        derived_updates.update_multires_body_meshes('server', 'uuid', 'segmentation', config)
    assert not [c for c in calls if c[0] == 'receipt']
