"""Plot actual candidate counts and compile/render an IEEE toolchain fixture."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / '.cache/course_fast_track/plotting'
sys.path.insert(0, str(OVERLAY))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/course_fast_track/matplotlib'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'report_toolchain.json').exists():
        raise SystemExit('Preserve prior attempt')
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt

    work = ROOT / '.cache/course_fast_track/tooling'
    work.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'data/candidates/v5/sandbox_pairs.jsonl'
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    labels = ('reversible', 'partially_reversible', 'irreversible')
    counts = {str(h): dict(Counter(r['labels_by_horizon'][str(h)]['label'] for r in rows))
              for h in (1, 2, 3)}
    assert len(rows) == 1800 and all(sum(x.values()) == len(rows) for x in counts.values())
    fig, ax = plt.subplots(figsize=(3.35, 2.15), layout='constrained')
    bottom = [0, 0, 0]
    for label, display, color in zip(labels, ('Reversible', 'Partial', 'Irreversible'),
                                      ('#4775a1', '#daa34a', '#a85858')):
        values = [counts[str(h)].get(label, 0) for h in (1, 2, 3)]
        ax.bar([1, 2, 3], values, bottom=bottom, label=display, color=color, width=.65)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set(xlabel='Recovery horizon (additional actions)', ylabel='Examples', xticks=[1, 2, 3], ylim=(0, 2050))
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(.5, 1.18), ncol=3, frameon=False)
    fig.savefig(work / 'horizon_counts.pdf')
    fig.savefig(work / 'horizon_counts.png', dpi=160)
    plt.close(fig)
    lines = '\n'.join(f"{h} & {counts[str(h)]['reversible']} & {counts[str(h)]['partially_reversible']} & {counts[str(h)]['irreversible']} \\\\"
                      for h in (1, 2, 3))
    tex = r'''\documentclass[conference]{IEEEtran}
\usepackage{graphicx,booktabs}
\begin{document}
\title{Phase I Toolchain Check}
\author{\IEEEauthorblockN{Automated course-project tooling verification}}
\maketitle
\begin{abstract}
This test verifies plotting, two-column compilation and PDF rendering.
The values are computed from existing automatically labeled sandbox candidates.
No pretrained-model result or independent human judgment is represented.
\end{abstract}
\section{Dataset check}
The source contains 1,800 sandbox examples. Each is classified at one, two
and three additional recovery actions. The counts below are read directly
from the saved labels, with no model inference.
\begin{table}[h]\centering
\caption{Automatically derived example counts.}
\begin{tabular}{rrrr}\toprule
Horizon & Reversible & Partial & Irreversible\\\midrule
''' + lines + r'''
\bottomrule\end{tabular}\end{table}
\begin{figure}[h]\centering
\includegraphics[width=\columnwidth]{horizon_counts.pdf}
\caption{The same dataset counts by recovery horizon. This is a data
distribution plot, not model accuracy.}\end{figure}
\newpage
\section{Scope of this artifact}
The IEEE conference class, plot generation and document compilation are
tested here. Passing this check does not complete the final report,
the experimental phase or the original assignment requirements.
Human annotation and certified physical-world coverage remain absent.
\section{Reproduction}
The accompanying script saves the dataset hash, plotting versions and
compiler/rendering receipts. The final report will use actual experiment
records and will disclose any reduced course-project scope.
\end{document}
'''
    (work / 'tooling_smoke.tex').write_text(tex)
    command = ['pdflatex', '-no-shell-escape', '-halt-on-error', '-interaction=nonstopmode', 'tooling_smoke.tex']
    result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=60)
    (args.output_dir / 'latex_compile.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise SystemExit('LaTeX failed; preserve log')
    pdf = args.output_dir / 'tooling_smoke.pdf'
    shutil.copy2(work / 'tooling_smoke.pdf', pdf)
    shutil.copy2(work / 'tooling_smoke.tex', args.output_dir / 'tooling_smoke.tex')
    shutil.copy2(work / 'horizon_counts.pdf', args.output_dir / 'horizon_counts.pdf')
    shutil.copy2(work / 'horizon_counts.png', args.output_dir / 'horizon_counts.png')
    info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True, check=True).stdout
    (args.output_dir / 'pdfinfo.txt').write_text(info)
    fonts = work / 'fonts.conf'
    font_cache = work / 'font-cache'
    font_cache.mkdir(exist_ok=True)
    fonts.write_text('<?xml version="1.0"?><fontconfig><dir>/System/Library/Fonts</dir>'
                     '<dir>/System/Library/Fonts/Supplemental</dir>'
                     f'<cachedir>{font_cache}</cachedir></fontconfig>')
    render_env = {**os.environ, 'FONTCONFIG_FILE': str(fonts), 'FONTCONFIG_PATH': str(work),
                  'XDG_CACHE_HOME': str(work)}
    render = subprocess.run(['pdftoppm', '-f', '1', '-l', '1', '-r', '110', '-singlefile', '-png',
                             str(pdf), str(work / 'preview')], env=render_env,
                            capture_output=True, text=True, timeout=60)
    (args.output_dir / 'render.log').write_text(render.stdout + render.stderr)
    assert render.returncode == 0 and (work / 'preview.png').exists()
    report = {'scope': 'IEEE_compile_and_plot_smoke_not_final_paper_or_model_result',
              'status': 'PASS', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dataset_sha256': hashlib.sha256(data.read_bytes()).hexdigest(), 'n': len(rows),
              'class_counts_by_horizon': counts,
              'overlay_packages': {p: importlib.metadata.version(p) for p in (
                  'matplotlib', 'contourpy', 'cycler', 'fonttools', 'kiwisolver', 'pillow',
                  'pyparsing', 'python-dateutil', 'six', 'numpy', 'packaging')},
              'latex_return_code': result.returncode, 'render_return_code': render.returncode,
              'render_preview_path': str(work / 'preview.png'), 'visual_QA_status': 'PENDING',
              'outputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (pdf, args.output_dir / 'horizon_counts.pdf', args.output_dir / 'tooling_smoke.tex')}}
    (args.output_dir / 'report_toolchain.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: actual count plot, IEEE compilation and PNG render; visual review pending')


if __name__ == '__main__':
    main()
