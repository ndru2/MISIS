import os

from pdfscan.env import load_project_env


def test_project_env_loads_without_overwriting_process_environment(tmp_path, monkeypatch):
    (tmp_path / '.env').write_text('RAG_LLM_MODEL=hosted-model\nRAG_LLM_API_KEY=test-token\n')
    monkeypatch.setenv('PDFSCAN_ROOT', str(tmp_path))
    monkeypatch.setenv('RAG_LLM_MODEL', 'deployment-model')
    monkeypatch.delenv('RAG_LLM_API_KEY', raising=False)
    load_project_env()
    assert os.environ['RAG_LLM_MODEL'] == 'deployment-model'
    assert os.environ['RAG_LLM_API_KEY'] == 'test-token'
