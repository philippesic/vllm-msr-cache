"""Failure contracts for the resumable campaign, independent of CUDA."""
import json
from pathlib import Path

import pytest

from benchmarks.serving_suite.campaign import catalogs, compile_campaign, order
from benchmarks.serving_suite.campaign_contracts import (
    Ledger, atomic_json, digest, load, local_path, seal_tree, validate_campaign,
    validate_host, verify_tree,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'benchmarks/serving_suite/campaigns/stage6.json'


def test_stage6_catalog_matrix():
    spec, records = catalogs(CAMPAIGN)
    assert sum(len(s) for m in spec['matrix'].values() for s in m.values()) == 54
    assert len(spec['matrix']['qwen3_8b_awq']) == 12
    assert len(records['models']) == 3
    assert records['policy']['experimental_overrides'] == {}


def test_critical_settings_rejected():
    spec, _ = catalogs(CAMPAIGN)
    spec['execution'] = 'graphs'
    with pytest.raises(ValueError, match='eager'):
        validate_campaign(spec)
    spec['execution'] = 'eager'
    spec['surprise'] = True
    with pytest.raises(ValueError, match='unknown'):
        validate_campaign(spec)


def test_compile_reports_unresolved_without_gpu(tmp_path):
    host = load(ROOT / 'benchmarks/serving_suite/templates/host.example.json')
    host['asset_cache'] = str(tmp_path / 'cache')
    path = tmp_path / 'host.json'
    atomic_json(path, host)
    result = compile_campaign(CAMPAIGN, path, tmp_path / 'run')
    assert not result['runnable']
    assert result['configurations_per_repeat'] == 54
    assert any('model preparation' in text for text in result['unresolved'])
    with pytest.raises(ValueError, match='already compiled'):
        compile_campaign(CAMPAIGN, path, tmp_path / 'run')


def test_json_duplicate_and_nonfinite_rejected(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{"budget": 1, "budget": 2}')
    with pytest.raises(ValueError, match='Duplicate'):
        load(path)
    path.write_text('{"budget": NaN}')
    with pytest.raises(ValueError, match='Nonfinite'):
        load(path)


def test_path_traversal_and_symlink_escape(tmp_path):
    for name in ('../secret', '/etc/passwd', 'a/../../secret'):
        with pytest.raises(ValueError):
            local_path(tmp_path, name)
    (tmp_path / 'link').symlink_to('/tmp')
    with pytest.raises(ValueError):
        local_path(tmp_path, 'link/escaped')


def test_seal_requires_all_files_and_original_identity(tmp_path):
    atomic_json(tmp_path / 'value.json', {'a': 1})
    seal = seal_tree(tmp_path)
    assert verify_tree(tmp_path, seal) == 1
    (tmp_path / 'unsealed.txt').write_text('new')
    with pytest.raises(ValueError, match='unsealed'):
        verify_tree(tmp_path, seal)
    (tmp_path / 'unsealed.txt').unlink()
    atomic_json(tmp_path / 'value.json', {'a': 2})
    with pytest.raises(ValueError, match='Changed artifact'):
        verify_tree(tmp_path, seal)
    seal_tree(tmp_path)
    with pytest.raises(ValueError, match='Seal identity'):
        verify_tree(tmp_path, seal)


def test_four_arm_balance():
    arms = ['arc', 'wait', 'continuum', 'upstream']
    rows = [order(arms, r, 17011) for r in range(4)]
    for position in range(4):
        assert {row[position] for row in rows} == set(arms)
    edges = [(a, b) for row in rows for a, b in zip(row, row[1:])]
    assert len(set(edges)) == 12
    assert order(arms[:2], 0, 17011) == order(arms[:2], 1, 17011)[::-1]
