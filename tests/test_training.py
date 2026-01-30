import os
import json
from pathlib import Path

def test_train_creates_model(tmp_path):
    from scripts.train_adaptive import main as train_main
    out = tmp_path / 'model_params.json'
    # run training script (it writes default model)
    import sys
    sys_argv = __import__('sys').argv
    try:
        __import__('sys').argv = ['train_adaptive', '--out', str(out)]
        train_main()
    finally:
        __import__('sys').argv = sys_argv
    assert out.exists()
    data = json.loads(out.read_text(encoding='utf-8'))
    assert 'integrated_multiplier' in data