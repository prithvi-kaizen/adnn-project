"""Locate action-overlap tokens using character offsets, without truncation."""


def action_positions(offsets, attention, start: int, end: int) -> list[int]:
    if not 0 <= start < end or len(offsets) != len(attention):
        raise ValueError('Invalid action span or attention length')
    selected, covered = [], set()
    for i, ((left, right), valid) in enumerate(zip(offsets, attention)):
        if not valid or right <= left:
            continue
        if left < end and right > start:
            selected.append(i)
            covered.update(range(max(left, start), min(right, end)))
    if not selected or covered != set(range(start, end)):
        raise ValueError('Action absent, normalized away or truncated; never substitute a delimiter')
    return selected
