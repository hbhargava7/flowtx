import pytest

from flowtx.flowengine import FlowEngine

def test_flowengine_init_fromfile():
    engine = FlowEngine(wsp_path='tests/data/test_workspace.wsp', use_cache=False)
    assert len(engine.wsp.get_sample_ids()) == 10
