"""Execute only the fixed eighteen-call exploratory format diagnostic."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_exploratory_ollama import parser_diagnostic  # noqa: E402
from src.agent.loop import initial_messages  # noqa: E402
from src.data.pilot import scenarios  # noqa: E402
from src.models.ollama_dev import (  # noqa: E402
    OPTIONS,
    OllamaDevelopmentClient,
    ResourceMonitor,
    api,
    append_jsonl,
    render,
    utc,
)

ARGUMENTS = {'read_file': ('path',), 'edit_file': ('path', 'content'), 'delete_file': ('path',),
             'restore_file': ('path', 'source'), 'make_backup': ('path',), 'remove_backup': ('path',),
             'send_email': ('message_id',), 'cancel_email': ('message_id',),
             'delete_email_record': ('message_id',), 'transfer_money': ('transfer_id',),
             'rollback_transfer': ('transfer_id',), 'settle_transfer': ('transfer_id',),
             'check_state': (), 'ask_user': (), 'wait': (), 'finish': ('reason',), 'refuse': ('reason',)}
SCHEMA = {'anyOf': [{'type': 'object', 'properties': {'tool': {'const': tool},
            'arguments': {'type': 'object', 'properties': {key: {'type': 'string'} for key in keys},
                          'required': list(keys), 'additionalProperties': False}},
            'required': ['tool', 'arguments'], 'additionalProperties': False}
           for tool, keys in ARGUMENTS.items()]}
EXTRA = '\nExample of the required envelope: {"tool":"check_state","arguments":{}}. ' \
        'Each response must be one tool call, never the whole state or a function expression. ' \
        'Use the exact argument names in this schema:\n' + json.dumps(SCHEMA, separators=(',', ':'))
CANARY = {'type': 'object', 'properties': {'probe': {'const': 'ok'}},
          'required': ['probe'], 'additionalProperties': False}


def main():
    out = ROOT / 'artifacts/exploratory/ollama_diagnostic_v2'
    if out.exists():
        raise SystemExit('Preserve prior attempts; no adaptive rerun')
    out.mkdir(parents=True)
    client = OllamaDevelopmentClient('1b', out)
    client.deadline = time.monotonic() + 300
    meta = {'scope': 'exploratory_not_BF16', 'started_at_utc': utc(),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
            'options': OPTIONS, 'assigned_fixture_calls': 16, 'assigned_canaries': 2,
            'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path(__file__).resolve(), ROOT / 'src/models/ollama_dev.py',
                              ROOT / 'src/agent/loop.py', ROOT / 'docs/OLLAMA_DIAGNOSTIC_v2_PROTOCOL.md')}}
    try:
        meta['preflight'] = client.preflight()
        tag = api('/api/show', {'model': client.model})
        meta['tag_template'] = tag.get('template')
        (out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        workload = []
        for index in (0, 2, 4, 6):
            scenario = scenarios()[index]
            for condition in ('original_raw', 'explicit_raw', 'explicit_raw_schema', 'explicit_chat_schema'):
                messages = initial_messages(scenario)
                if condition != 'original_raw':
                    messages[0]['content'] += EXTRA
                workload.append((scenario, condition, messages))
        canary_messages = [{'role': 'user', 'content': 'Return exactly {"probe":"wrong"}.'}]
        workload += [(None, name, canary_messages) for name in ('canary_raw', 'canary_chat')]
        with ResourceMonitor(out / 'resources.jsonl'):
            for scenario, condition, messages in workload:
                client.guard()
                chat = 'chat' in condition
                schema = CANARY if scenario is None else SCHEMA
                payload = {'model': client.model, 'options': OPTIONS, 'stream': False, 'keep_alive': '5m'}
                if chat:
                    payload['messages'] = messages
                else:
                    payload.update(raw=True, prompt=render(messages))
                if 'schema' in condition or scenario is None:
                    payload['format'] = schema
                if len(render(messages).encode()) + 130 > OPTIONS['num_ctx']:
                    raise ValueError('Conservative byte context bound failed')
                row = {'condition': condition, 'scenario_id': scenario.scenario_id if scenario else None,
                       'template_id': scenario.template_id if scenario else None,
                       'endpoint': '/api/chat' if chat else '/api/generate', 'request': payload,
                       'started_at_utc': utc()}
                append_jsonl(out / 'requests.jsonl', {'event': 'started', **row})
                started = time.monotonic()
                try:
                    client.owns_residency = True
                    response = api(row['endpoint'], payload)
                    text = response.get('message', {}).get('content', '') if chat else response.get('response', '')
                    row.update(response=response, wall_seconds=time.monotonic() - started, status='completed', text=text)
                    if scenario:
                        row['parser'] = parser_diagnostic(scenario, text)
                    else:
                        try:
                            row['canary_compliant'] = json.loads(text) == {'probe': 'ok'}
                        except ValueError:
                            row['canary_compliant'] = False
                except Exception as error:
                    row.update(status='request_failed', error_type=type(error).__name__, error=str(error),
                               wall_seconds=time.monotonic() - started)
                append_jsonl(out / 'outcomes.jsonl', row)
                print(json.dumps({k: row[k] for k in ('condition', 'scenario_id', 'status')}), flush=True)
                client.guard()
        if not any(m['digest'] == client.digest and m['name'] == client.model for m in api('/api/tags')['models']):
            raise ValueError('Model digest changed')
        meta['status'] = 'completed'
    except Exception as error:
        meta.update(status='stopped', error_type=type(error).__name__, error=str(error))
    finally:
        meta['ended_at_utc'] = utc()
        meta['residents_after_unload'] = client.unload()
        (out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'status': meta['status']}, indent=2))


if __name__ == '__main__':
    main()
