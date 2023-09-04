import pytest

from flowtx.flowengine import FlowEngine

def test_flowengine_init_from_file():
    engine = FlowEngine(wsp_path='tests/data/test_workspace.wsp', use_cache=False)
    assert len(engine.wsp.get_sample_ids()) == 5

def test_flowengine_init_from_cache():
    engine = FlowEngine(use_cache=True, cache_path='tests/data/wsp_cache.pkl')
    assert len(engine.wsp.get_sample_ids()) == 5