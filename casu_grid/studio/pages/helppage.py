"""Help: a three-minute quick start, the glossary behind every (i) button, troubleshooting and settings."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

from .. import help as helptext
from ..widgets import Card, PageHeader, make_label, page_container, scroll_page

QUICK_START = [
    ("Look at the Overview", "It tells you whether the simulator, GPU and memory are ready and what to do next. If no learning run exists, it says so plainly."),
    ("Open Experiments", "Pick “XY1 Point Navigation · Mini World”. The grey 🔒 block shows what is fixed (world, observation, actions, reward, PPO settings). "
                         "Below it you choose the workload: the default is the validated 64 worlds × 8 workers on the GPU."),
    ("Choose a budget and validate", "“Smoke test” checks the whole pipeline in about two minutes. “Diagnostic” collects about a million decisions (about ten minutes). "
                                     "The validation list turns green when everything needed is in place and explains every problem in words."),
    ("Start training", "Press Start. The Training page shows the real state of the process, the speed, the checkpoint, memory and whether goals reached are rising."),
    ("Read the curves", "Metrics has separate interactive charts: hover for exact values, scroll to zoom, drag to pan, double-click to reset, ⤢ to inspect one chart in detail."),
    ("Check it on unseen terrains", "Evaluation scores a checkpoint and the untrained baseline on held-out terrains with identical goals. This, not the training reward, tells you whether it learned."),
    ("Continue or compare", "Runs & checkpoints lists every run. Select a run and a checkpoint to continue it (compatibility is checked first), or select several runs to compare them."),
]

TROUBLE = [
    ("Start is greyed out", "Hover it: the tooltip names the first blocking problem, and the red ✖ lines in the validation list explain each one."),
    ("“Simulator processes are already running”", "A previous run, an evaluation, or a crashed run left processes behind. Stop it from the Training page; if nothing is listed there, check for leftover processes (python3 -m casu_grid.isaac_shard_worker)."),
    ("The first update takes long", "Isaac Sim starts and builds 64 worlds first (about a minute); the Training page says which phase it is in."),
    ("A run failed", "Open the run on the Training page: it names the cause in plain words, shows the technical text, and tells you whether a checkpoint survived and can be resumed."),
    ("A checkpoint cannot be resumed", "Its learning contract, world count or worker layout differs from the selected experiment. The Runs page says which."),
    ("Charts look empty", "Each empty chart explains why (for example “no episode has finished yet”). Gaps mean no sample, never zero."),
]


class HelpPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content, layout = page_container()
        outer.addWidget(scroll_page(content))
        layout.addWidget(PageHeader("Help", "Short answers where you need them; every (i) in the application opens the matching entry below."))
        quick = Card("Quick start: from opening the Studio to a first learning signal")
        text = "".join(f"<p><b>{i}. {title}.</b> {body}</p>" for i, (title, body) in enumerate(QUICK_START, 1))
        quick.add(make_label(text))
        layout.addWidget(quick)
        glossary = Card("Glossary")
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search the glossary…")
        self.search.textChanged.connect(self._filter)
        glossary.add(self.search)
        self.entries = []
        for topic in helptext.TOPICS.values():
            label = make_label(f"<b>{topic.title}</b><br>{topic.short}<br><span class='muted'>{topic.body}</span>")
            label.setProperty("search", f"{topic.title} {topic.short} {topic.body}".lower())
            glossary.add(label)
            self.entries.append(label)
        layout.addWidget(glossary)
        trouble = Card("When something does not work")
        trouble.add(make_label("".join(f"<p><b>{t}.</b> {b}</p>" for t, b in TROUBLE)))
        layout.addWidget(trouble)
        settings = Card("Settings")
        form = QtWidgets.QGridLayout()
        self.isaac = QtWidgets.QLineEdit(str(ctx.install.sim_root or ""))
        self.isaac.setPlaceholderText("Isaac Sim folder (contains python.sh)")
        self.runs = QtWidgets.QLineEdit(str(ctx.run_root))
        form.addWidget(make_label("Isaac Sim folder", wrap=False), 0, 0)
        form.addWidget(self.isaac, 0, 1)
        form.addWidget(make_label("Run folder", wrap=False), 1, 0)
        form.addWidget(self.runs, 1, 1)
        form.setColumnStretch(1, 1)
        settings.add(form)
        apply_ = QtWidgets.QPushButton("Apply")
        apply_.clicked.connect(self._apply)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(apply_)
        row.addStretch(1)
        settings.add(row)
        self.settings_note = make_label("", "hint")
        settings.add(self.settings_note)
        layout.addWidget(settings)
        layout.addStretch(1)

    def _filter(self, text: str):
        needle = text.strip().lower()
        for label in self.entries:
            label.setVisible(not needle or needle in label.property("search"))

    def _apply(self):
        self.ctx.set_isaac(self.isaac.text().strip())
        self.ctx.set_run_root(Path(self.runs.text().strip()).expanduser())
        ok = self.ctx.install.ok
        self.settings_note.setText("Isaac Sim found." if ok else "Isaac Sim was not found: " + "; ".join(self.ctx.install.problems))
