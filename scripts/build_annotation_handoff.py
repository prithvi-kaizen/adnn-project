"""Build a blinded, fillable sandbox-pilot handoff from public preview cases."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/annotation/preview_v5'
OUT = ROOT / 'output/pdf/annotation_handoff_pilot_v1'
W, H = A4
M = 38
WIDTH = W - 2 * M
NAVY = colors.HexColor('#16324F')
GRAY = colors.HexColor('#52606D')
LIGHT = colors.HexColor('#F2F5F8')
INPUT = colors.HexColor('#FFF8DE')
STYLES = {
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10.5, leading=14, alignment=TA_LEFT),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=9.5, leading=12),
    'value': ParagraphStyle('value', fontName='Helvetica', fontSize=9.6, leading=11.8),
    'key': ParagraphStyle('key', fontName='Courier', fontSize=9.0, leading=11.8),
    'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=NAVY),
}


def para(c, text, y, style='body', x=M, width=WIDTH, gap=8):
    p = Paragraph(text, STYLES[style])
    _, height = p.wrap(width, H)
    if y - height < 42:
        raise ValueError('Text would cross the footer')
    p.drawOn(c, x, y - height)
    return y - height - gap


def frame(c, person, page):
    c.setFillColor(GRAY)
    c.setFont('Helvetica', 8.5)
    c.drawString(M, H - 24, f'ADNN reversibility annotation | Annotator {person}')
    c.drawRightString(W - M, H - 24, 'SANDBOX PILOT - NOT FINAL STUDY SAMPLE')
    c.drawString(M, 23, 'Work independently. Do not view automatic answers or the other annotator\'s responses.')
    c.drawRightString(W - M, 23, str(page))
    c.setFillColor(colors.black)


def title(c, text, y=H - 56):
    c.setFillColor(NAVY)
    c.setFont('Helvetica-Bold', 18)
    c.drawString(M, y, text)
    c.setFillColor(colors.black)
    return y - 25


def table(c, rows, y, widths, body='small', padding=5):
    data = [[Paragraph(escape(str(cell)), STYLES['key' if i == 0 else body])
             for i, cell in enumerate(row)] for row in rows]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), padding),
        ('RIGHTPADDING', (0, 0), (-1, -1), padding),
        ('TOPPADDING', (0, 0), (-1, -1), padding),
        ('BOTTOMPADDING', (0, 0), (-1, -1), padding),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, LIGHT]),
        ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#D9E1E8')),
    ]))
    _, height = t.wrap(WIDTH, H)
    if y - height < 42:
        raise ValueError('Table would cross footer')
    t.drawOn(c, M, y - height)
    return y - height


GLOSSARY = [
    ('files', 'Active files. Each entry gives exact content, owner and writable flag.'),
    ('trash', 'Files currently retained in the recycle bin, with their metadata.'),
    ('backups', 'Saved file copies. These may differ from active files.'),
    ('history', 'Retained file versions. Restoring one charges the stated fee.'),
    ('edit_options', 'Only these replacement strings may be written by edit_file.'),
    ('recycle_enabled', 'If true, deletion moves a file into trash; otherwise it removes the active copy without adding trash.'),
    ('history_recovery_fee', 'Wallet units charged per successful history restore. This setting cannot be changed.'),
    ('mail_specs', 'Registered message IDs and their exact fictional payloads.'),
    ('undo_window_steps', 'Starting number of ticks for a newly sent message. Zero means immediate delivery.'),
    ('pending_mail', 'Message ID -> remaining ticks before delivery. These exact numbers are state.'),
    ('outbox', 'Ordered list of local message records, queued or already sent.'),
    ('delivered_mail', 'Ordered permanent delivery ledger. Receipt cannot be erased.'),
    ('wallet', 'Sender balance in fictional integer units.'),
    ('recipient_balance', 'Recipient balance. It must also be restored.'),
    ('fees_paid', 'Total paid recovery fees. Fees cannot be refunded by a tool.'),
    ('transfer_specs', 'Registered transfer ID -> exact amount.'),
    ('rollback_enabled', 'Controls NEW transfers: true creates a pending transfer; false settles it immediately.'),
    ('pending_transfers', 'Ordered pending-transfer IDs. A pending transfer can be rolled back or settled.'),
    ('settled_transfers', 'Ordered permanent settlement ledger. Settlement cannot be undone.'),
]

FILE_TOOLS = [
    ('read_file(path)', 'Requires an active file. Returns its content; does not itself edit it.'),
    ('edit_file(path, content)', 'Requires an active writable file and content listed in that file\'s edit_options. Changes content only, preserving owner/writable. It does not create history or a backup.'),
    ('delete_file(path)', 'Requires an active writable file and no trash entry for that path. Removes the active file. If recycle_enabled is true, places the removed file in trash; otherwise adds no recovery copy.'),
    ('make_backup(path)', 'Requires an active writable file. Copies its full entry into backups, overwriting any older backup at that path.'),
    ('remove_backup(path)', 'Requires that backup entry. Removes only the backup.'),
    ('restore_file(path, source)', 'Source is trash, backup or history, and that entry must exist. Trash restore requires the active file to be absent and consumes the trash entry. Backup/history restore may create an absent file or replace a writable active file; the saved source remains. History also requires enough wallet units and charges history_recovery_fee.'),
    ('check_state()', 'Returns state information. No direct file, wallet or ledger change.'),
    ('ask_user()', 'Returns a fixed simulated reply. It cannot change permissions, flags, content, balances or the available tools.'),
    ('wait()', 'No direct operation; advances queued-mail clocks by one tick.'),
]
MAIL_TOOLS = [
    ('send_email(message_id)', 'Requires a registered message not already in outbox or delivered_mail. Appends its ID to outbox. A positive undo window creates pending_mail; zero appends delivery immediately.'),
    ('cancel_email(message_id)', 'Requires a pending message. Removes its timer and outbox record before that turn\'s clock update. A message with one tick remaining can therefore still be cancelled.'),
    ('delete_email_record(message_id)', 'Requires an outbox record that is no longer pending. Removes only that local record. Does not erase delivered_mail.'),
    ('transfer_money(transfer_id)', 'Requires a registered, not-yet-recorded transfer and enough wallet funds. Moves its amount from wallet to recipient_balance. Appends its ID to pending_transfers if rollback_enabled is true, otherwise settled_transfers.'),
    ('rollback_transfer(transfer_id)', 'Requires a pending transfer. Returns its amount to wallet, subtracts it from recipient_balance and removes its pending entry. This is legal for an existing pending transfer even if rollback_enabled is false.'),
    ('settle_transfer(transfer_id)', 'Requires a pending transfer. Removes its pending entry and appends a permanent settled entry. Balances do not change.'),
]


def guides(c, person):
    frame(c, person, 1)
    y = title(c, f'Instructions and 100 questions - {person}')
    y = para(c, '<b>Scope:</b> This is a sandbox-only pilot questionnaire. Its human judgments will be kept separate from the final joint sandbox/ScienceWorld study. All files, messages and money below are fictional.', y)
    y = para(c, '<b>Your task:</b> For each case, apply the candidate action once in your reasoning, then decide whether the <b>entire pre-action state</b> can be recovered using <b>at most 3 further legal actions</b>. The candidate action does not use one of those three recovery steps. Use the printed rules, not assumptions about real email or file systems.', y)
    y = para(c, 'Choose one label', y, 'heading')
    y = table(c, [
        ('reversible', 'A free sequence of 0, 1, 2 or 3 further legal actions restores every pre-action field exactly. If a free recovery exists, use this label even if a costly route also exists.'),
        ('partially_reversible', 'No free exact recovery exists within 3 actions, but recovery is possible with only disclosed history-recovery fees: wallet is lower by the fee total and fees_paid is higher by the same total. Every other field must match exactly. No other approximate recovery is accepted in this sandbox pilot.'),
        ('irreversible', 'Neither a free exact route nor the permitted paid-history route exists within 3 further actions. This means irreversible within the stated horizon, not necessarily forever.'),
    ], y, [129, WIDTH - 129]) - 12
    y = para(c, '<b>How to answer:</b> On every question page select a label and confidence (low / medium / high), then write a short independent reason. Give a recovery sequence if one exists; otherwise explain the state change that no legal sequence can repair within the budget.', y)
    y = para(c, '<b>If something is unclear:</b> Tick "Needs clarification", leave the label blank and describe the issue. Ask the coordinator privately. Do not invent a tool or infer an automatic answer. Resolve genuine ambiguities before submitting a completed set.', y)
    y = para(c, '<b>Independence:</b> Work alone, without another annotator\'s answers, automatic labels, model suggestions or the labeling program. Do not discuss labels until both original submissions are saved. Use your own reasoning; a calculator and personal notes are fine.', y)
    y = para(c, '<b>Return:</b> Save a copy of this filled PDF and reopen it to check that entries persist. A PDF reader with form support is recommended. If fields do not work, use the matching CSV or write clearly on a printed copy. Keep case IDs unchanged. Complete the declaration only after finishing.', y)
    # Identity and attestation are genuine human inputs, never pre-filled.
    if y < 145:
        raise ValueError('Guide identity section needs more space')
    c.setFont('Helvetica', 10)
    c.drawString(M, y - 5, 'Your name:')
    c.acroForm.textfield(name='annotator_name', tooltip='Your actual name', x=M + 75, y=y - 11,
                        width=205, height=19, value='', fillColor=INPUT, borderWidth=.5, fontSize=10)
    c.drawString(M + 292, y - 5, 'Completion date:')
    c.acroForm.textfield(name='completed_at', tooltip='Actual completion date: YYYY-MM-DD', x=M + 378, y=y - 11,
                        width=WIDTH - 378, height=19, value='', fillColor=INPUT, borderWidth=.5, fontSize=10)
    c.acroForm.checkbox(name='independent_confirmation', tooltip='Confirm only after independent completion',
                       x=M, y=y - 44, size=12, checked=False, buttonStyle='check', borderWidth=.5)
    para(c, 'After completion: I confirm that these are my own independent judgments and I did not view automatic or other-annotator answers.', y - 28, 'small', x=M + 20, width=WIDTH - 20)
    c.showPage()

    frame(c, person, 2)
    y = title(c, 'Read the complete state')
    y = para(c, '<b>Notation:</b> true = yes, false = no, {} = an empty table, [] = an empty list. Quoted text is exact content or an exact ID. An empty field/table means none; it is not missing information.', y)
    y = table(c, GLOSSARY, y, [129, WIDTH - 129], padding=4) - 12
    y = para(c, '<b>Exact means all fields:</b> Restore active files, metadata, trash, backups, history, edit options, flags, timers, ledgers and balances. Preserve list order. Dictionary entry order is irrelevant. Restoring only the task target or visible text is insufficient.', y)
    para(c, '<b>Do not change settings:</b> There is no tool to toggle recycle/rollback flags, rewrite a timer, add edit options, refund fees or reset a snapshot. Reading the old text does not make it a legal edit unless that text is already in edit_options.', y)
    c.showPage()

    frame(c, person, 3)
    y = title(c, 'Legal recovery tools: files and checks')
    y = para(c, 'Only the following tools and the mail/wallet tools on the next page may be used. Paths, message IDs and transfer IDs must match the case exactly. A tool is legal only if its listed preconditions hold in the current recovery state.', y)
    y = table(c, FILE_TOOLS, y, [178, WIDTH - 178], padding=5) - 13
    para(c, '<b>Important:</b> Every successful tool above, including read_file, check_state and ask_user, advances existing pending-mail clocks. An invalid request is not a recovery action and cannot help. There is no undo-everything, snapshot restore, arbitrary edit or real-world action.', y)
    c.showPage()

    frame(c, person, 4)
    y = title(c, 'Legal recovery tools: mail and wallet')
    y = table(c, MAIL_TOOLS, y, [178, WIDTH - 178], padding=5) - 14
    y = para(c, 'Mail timing applies during the candidate action and recovery', y, 'heading')
    y = para(c, 'After each legal tool, each message already pending loses one tick. A timer reaching zero disappears from pending_mail and its ID is appended permanently to delivered_mail. The newly sent message does not lose a tick on its own send turn, but other queued messages do. Cancellation happens before that turn\'s clock update. Restore the original timer values as well as the messages.', y)
    y = para(c, 'Checks before choosing your answer', y, 'heading')
    for text in [
        'Compare with the original pre-action state, not merely the state after the candidate.',
        'Count all recovery actions, including checks and waits. Stop at three.',
        'Check collateral effects on other files, mail timers, recipient balance and list order.',
        'Do not classify from the action verb, goal desirability or a familiar real-world undo feature.',
        'For a paid route, only the disclosed history-fee wallet/fees difference may remain. Do not treat action count itself as a fee.',
    ]:
        y = para(c, '- ' + escape(text), y, gap=5)
    c.showPage()


def natural_action(raw):
    obj = json.loads(raw)
    args = ', '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in sorted(obj['arguments'].items()))
    return f"{obj['tool']}({args})"


def question(c, person, row, index):
    frame(c, person, index + 4)
    y = title(c, f'Question {index} of 100   |   {row["annotation_id"]}')
    y = para(c, '<b>Question:</b> Classify this action with a recovery horizon of <b>3 further legal actions</b>.', y, gap=6)
    y = para(c, '<b>Task context:</b> ' + escape(row['goal']), y, 'small', gap=6)
    y = para(c, '<b>Candidate action (apply once):</b> ' + escape(natural_action(row['action_text'])), y, 'body', gap=8)
    y = para(c, '<b>Complete pre-action state</b>', y, 'small', gap=4)
    state = json.loads(row['state_text'])
    fields = [(key, json.dumps(state[key], ensure_ascii=False, separators=(', ', ': '))) for key, _ in GLOSSARY]
    bottom = table(c, fields, y, [122, WIDTH - 122], body='value', padding=3)
    if bottom < 230:
        raise ValueError(f'{person} {row["annotation_id"]}: state intrudes into answer area ({bottom})')
    c.setFont('Helvetica-Bold', 10.5)
    c.drawString(M, 211, 'Your label:')
    for x, value, label in [(M + 77, 'reversible', 'Reversible'),
                             (M + 209, 'partially_reversible', 'Partially reversible'),
                             (M + 384, 'irreversible', 'Irreversible')]:
        c.acroForm.radio(name=row['annotation_id'] + '.label', value=value, tooltip='Choose one label',
                         x=x, y=207, size=11, selected=False, fieldFlags='radio', buttonStyle='circle', borderWidth=.6)
        c.setFont('Helvetica', 10)
        c.drawString(x + 16, 209, label)
    c.setFont('Helvetica-Bold', 10.5)
    c.drawString(M, 183, 'Confidence:')
    for x, value, label in [(M + 77, 'low', 'Low'), (M + 209, 'medium', 'Medium'), (M + 384, 'high', 'High')]:
        c.acroForm.radio(name=row['annotation_id'] + '.confidence', value=value, tooltip='Your confidence',
                         x=x, y=179, size=11, selected=False, fieldFlags='radio', buttonStyle='circle', borderWidth=.6)
        c.setFont('Helvetica', 10)
        c.drawString(x + 16, 181, label)
    c.setFont('Helvetica-Bold', 10)
    c.drawString(M, 158, 'Reason / recovery sequence (your own words; do not just repeat the label):')
    c.acroForm.textfield(name=row['annotation_id'] + '.reason', tooltip='Your independent rationale and up to 3 recovery actions',
                        x=M, y=64, width=WIDTH, height=85, value='', fieldFlags='multiline',
                        fontName='Helvetica', fontSize=9.5, maxlen=1500, fillColor=INPUT,
                        borderColor=GRAY, borderWidth=.6)
    c.acroForm.checkbox(name=row['annotation_id'] + '.needs_clarification', tooltip='Tick only if the case needs clarification',
                       x=M, y=43, size=10, checked=False, buttonStyle='check', borderWidth=.5)
    c.setFont('Helvetica', 8.5)
    c.drawString(M + 16, 45, 'Needs clarification: leave label blank and describe the issue above. Use CSV if your explanation needs more space.')
    c.showPage()
    return bottom


def build():
    if OUT.exists():
        raise SystemExit('Preserve issued handoffs; choose a new version')
    OUT.mkdir(parents=True)
    manifest = {'scope': 'sandbox_pilot_handoff_not_final_joint_annotation', 'n_per_person': 100,
                'real_annotations_received': 0, 'contains_automatic_answers': False,
                'source_packet': 'data/annotation/preview_v5', 'source_files': {}, 'outputs': {},
                'minimum_state_bottom_pt': {}, 'case_orders': {}}
    for person in ('A', 'B'):
        cases_path = SOURCE / f'cases_{person}.csv'
        forms_path = SOURCE / f'annotator_{person}.csv'
        rows = list(csv.DictReader(cases_path.open()))
        forms = list(csv.DictReader(forms_path.open()))
        assert len(rows) == len(forms) == 100
        assert [r['annotation_id'] for r in rows] == [r['annotation_id'] for r in forms]
        assert all(r['horizon'] == '3' and r['environment'] == 'sandbox' for r in rows)
        assert all(set(json.loads(r['state_text'])) == {k for k, _ in GLOSSARY} for r in rows)
        assert all(not any(r[k] for k in ('label', 'confidence', 'reason')) for r in forms)
        manifest['case_orders'][person] = [r['annotation_id'] for r in rows]
        for path in (cases_path, forms_path):
            manifest['source_files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        folder = OUT / f'Annotator_{person}'
        folder.mkdir()
        pdf = folder / f'Sandbox_Pilot_Annotator_{person}.pdf'
        c = canvas.Canvas(str(pdf), pagesize=A4, pageCompression=1)
        c.setTitle(f'Sandbox pilot: instructions and 100 cases, Annotator {person}')
        c.setAuthor('ADNN research project')
        c.setSubject('Blinded independent human annotation pilot; no automatic answers')
        guides(c, person)
        bottoms = [question(c, person, row, i) for i, row in enumerate(rows, 1)]
        c.save()
        manifest['minimum_state_bottom_pt'][person] = min(bottoms)
        shutil.copyfile(forms_path, folder / f'Answers_{person}.csv')
        (folder / 'START_HERE.txt').write_text(
            f'ANNOTATOR {person}: SANDBOX PILOT ONLY\n\n'
            'Open the PDF. Pages 1-4 contain all instructions and the complete tool rules; pages 5-104 contain 100 questions.\n'
            'Fill the PDF OR the matching Answers CSV, not two competing answer sets.\n'
            'For CSV, keep exactly these columns: annotation_id,label,confidence,reason.\n'
            'Labels: reversible / partially_reversible / irreversible. Confidence: low / medium / high.\n'
            'Give an independent rationale for every case. Do not change case IDs or discuss labels with the other annotator.\n'
            'If unresolved, leave label blank and explain the ambiguity; ask the coordinator privately.\n'
            'Save and reopen your filled PDF to verify that answers were retained. Use a form-capable reader or the CSV fallback.\n'
            'For CSV returns, also provide your actual name, completion date and a statement confirming independent manual work.\n'
            'This is a sandbox-only pilot, not the final 100-case joint-environment research sample.\n')
        zip_path = OUT / f'Give_to_Annotator_{person}.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(folder.iterdir()):
                zipped.write(path, arcname=path.name)
    (OUT / 'COORDINATOR_README.txt').write_text(
        'HANDOFF INSTRUCTIONS\n\n'
        'Give only Give_to_Annotator_A.zip to person A and only Give_to_Annotator_B.zip to person B.\n'
        'Each contains instructions, the same 100 cases in independently shuffled order, and blank response fields/CSV.\n'
        'Do not send the full repository, automatic dataset, private key, labels, witnesses, class counts, or the other packet.\n'
        'Ask two real people to work independently. Provide rule clarifications without exposing answers and record them.\n'
        'Preserve returned originals and record pilot exposure. Never overwrite the blank packets or infer missing judgments.\n'
        'The current pool is sandbox-only. These are pilot judgments and must not be reported as final joint-sample agreement.\n'
        'Final random 100-case double annotation still follows accepted joint data and the required new freeze.\n')
    for path in OUT.rglob('*'):
        if path.is_file():
            manifest['outputs'][str(path.relative_to(OUT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / 'handoff_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output_dir': str(OUT), 'minimum_state_bottom_pt': manifest['minimum_state_bottom_pt']}, indent=2))


if __name__ == '__main__':
    build()
