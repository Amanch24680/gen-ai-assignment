from unittest.mock import MagicMock, patch
import pytest
from app.config import settings
from app.judge import JudgeClient


def test_judge_client_independent_configuration():
    # Verify generator and judge configurations are separate
    assert hasattr(settings, "generator_model")
    assert hasattr(settings, "judge_model")
    assert settings.judge_model is not None

    client = JudgeClient(judge_model="custom_judge:7b", base_url="http://localhost:11434")
    assert client.judge_model == "custom_judge:7b"


@patch("httpx.Client.post")
def test_judge_client_mocked_ollama_call(mock_post, tmp_path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": '{"overall_score": 4.5}',
        "prompt_eval_count": 120,
        "eval_count": 45,
    }
    mock_post.return_value = mock_resp

    audit_path = tmp_path / "audit.log"
    client = JudgeClient(audit_log_path=audit_path)

    raw_text, latency = client.generate_judge_response("sys prompt", "user prompt")

    assert '{"overall_score": 4.5}' in raw_text
    assert latency >= 0.0
    assert client.total_calls == 1
    assert client.total_prompt_tokens == 120
    assert client.total_eval_tokens == 45
    assert audit_path.exists()
