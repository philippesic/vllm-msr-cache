"""Explicit preparation and content identity for public campaign assets.

Network libraries are imported only by prepare. Measurement must use verified
local snapshots. No tokens or process environment are serialized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign_contracts import atomic_json, file_hash, load, validate_campaign, validate_host
from .contracts import digest

MODEL_PATTERNS = ['*.json', '*.jinja', '*.model', '*.txt', '*.safetensors', 'LICENSE*', 'README.md', '*.tiktoken']


def model_identity(path):
    path = Path(path)
    if not (path / 'config.json').is_file():
        raise ValueError('Missing model config.json')
    files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*'))
             if p.is_file() and '.cache' not in p.relative_to(path).parts}
    if not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Model has no safetensors weights')
    config = load(path / 'config.json')
    text = config.get('text_config', config)
    head_dim = text.get('head_dim') or text['hidden_size'] // text['num_attention_heads']
    kv_bytes = text['num_hidden_layers'] * 2 * text.get('num_key_value_heads', text['num_attention_heads']) * head_dim * 2
    tokenizer_files = {k: v for k, v in files.items() if k.startswith(('tokenizer', 'special_tokens', 'chat_template', 'tekken')) or k.endswith('.model')}
    return dict(files=files, content_digest=digest(files), tokenizer_digest=digest(tokenizer_files),
                config=config, estimated_fp16_kv_bytes_per_token=kv_bytes,
                weight_bytes=sum((path / n).stat().st_size for n in files if n.endswith('.safetensors')))


def prepare(campaign_path, host_path, *, metadata_only=False):
    from huggingface_hub import HfApi, snapshot_download
    campaign = validate_campaign(load(campaign_path))
    host = validate_host(load(host_path))
    root = Path(campaign_path).resolve().parent
    catalog = load(root / campaign['models'])
    cache = Path(host['asset_cache'])
    cache.mkdir(parents=True, exist_ok=True)
    receipt_path = cache / (campaign['id'] + '-assets.json')
    receipt = load(receipt_path) if receipt_path.exists() else dict(schema='serving-campaign-assets-v1', models={})
    api = HfApi()
    for name in campaign['matrix']:
        item = catalog['models'][name]
        old = receipt['models'].get(name, {})
        revision = old.get('revision') or item.get('revision') or api.model_info(item['repository']).sha
        if old and old.get('repository') != item['repository']:
            raise ValueError('Prepared repository changed; use new asset identity')
        info = api.model_info(item['repository'], revision=revision)
        record = dict(repository=item['repository'], revision=revision,
                      license=(info.card_data.to_dict() if info.card_data else {}).get('license'),
                      source_files=[x.rfilename for x in info.siblings], status='resolved')
        receipt['models'][name] = {**old, **record}
        atomic_json(receipt_path, receipt)
        if metadata_only:
            continue
        print(f'Preparing {name} at immutable revision {revision}', flush=True)
        snapshot = snapshot_download(item['repository'], revision=revision,
                                     allow_patterns=MODEL_PATTERNS, max_workers=2)
        identity = model_identity(snapshot)
        receipt['models'][name].update(path=snapshot, identity=identity, status='prepared')
        atomic_json(receipt_path, receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--host', type=Path, required=True)
    p.add_argument('--metadata-only', action='store_true')
    a = p.parse_args()
    result = prepare(a.campaign, a.host, metadata_only=a.metadata_only)
    print(json.dumps({k: {n: v for n, v in record.items() if n not in ('identity', 'source_files')}
                      for k, record in result['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
