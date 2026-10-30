"""
Tests for resolve_ref(), with DVID's repo info faked.
"""
import pytest

from neuclease.dvid import repo

OLD = 'a' * 32    # An older node on the master branch
HEAD = 'b' * 32   # The head of the master branch
SIDE = 'c' * 32   # The head of a side branch, 'side'


@pytest.fixture
def fake_repo(monkeypatch):
    def fetch_repo_info(server, uuid, *, session=None):
        # Like DVID, accept any ref which starts with a known uuid
        # (including '<uuid>:<branch>').
        if not any(uuid.startswith(u[:len(uuid)]) or uuid.startswith(u) for u in (OLD, HEAD, SIDE)):
            raise RuntimeError(f"No matching uuid for '{uuid}'")
        return {'DAG': {'Nodes': {OLD: {}, HEAD: {}, SIDE: {}}}}

    def find_branch_nodes(server, repo_uuid=None, branch=None, *args, **kwargs):
        # (Any node, possibly abbreviated, identifies the repo.)
        assert not repo_uuid or any(u.startswith(repo_uuid) for u in (OLD, HEAD, SIDE))
        return {'': [OLD, HEAD], 'side': [OLD, SIDE]}[branch]

    monkeypatch.setattr(repo, 'fetch_repo_info', fetch_repo_info)
    monkeypatch.setattr(repo, 'find_branch_nodes', find_branch_nodes)


def test_resolve_uuid(fake_repo):
    assert repo.resolve_ref('server', OLD) == OLD
    assert repo.resolve_ref('server', OLD[:6]) == OLD[:6]
    assert repo.resolve_ref('server', OLD[:6], expand=True) == OLD


def test_resolve_branch(fake_repo):
    assert repo.resolve_ref('server', 'master') == HEAD
    assert repo.resolve_ref('server', ':master') == HEAD
    assert repo.resolve_ref('server', 'master~1') == OLD


def test_resolve_uuid_and_branch(fake_repo):
    """'<uuid>:<branch>' is the branch's head, even if <uuid> is a complete uuid of another node."""
    assert repo.resolve_ref('server', f'{OLD[:6]}:master', expand=True) == HEAD
    assert repo.resolve_ref('server', f'{OLD}:master', expand=True) == HEAD
    assert repo.resolve_ref('server', f'{OLD}:side', expand=True) == SIDE
    assert repo.resolve_ref('server', f'{HEAD}:master', expand=True) == HEAD


def test_is_branch_head(fake_repo, monkeypatch):
    pytest.importorskip("vol2mesh")
    from neuclease.misc import multiresmesh
    monkeypatch.setattr(multiresmesh, 'resolve_ref', repo.resolve_ref)
    assert multiresmesh._is_branch_head('server', HEAD, 'master')
    assert not multiresmesh._is_branch_head('server', OLD, 'master')
    assert multiresmesh._is_branch_head('server', SIDE, 'side')
    assert not multiresmesh._is_branch_head('server', SIDE, 'master')
