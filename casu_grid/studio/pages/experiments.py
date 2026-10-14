"""Experiments: select an experiment, review what is fixed, choose the workload, validate, start."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import launch, theme
from ..experiments import RECOMMENDED, ExperimentInfo
from ..runs import CheckpointInfo, RunRecord, human_bytes, human_count, human_duration
from ..widgets import Banner, Card, InfoButton, PageHeader, clear_layout, make_label, page_container, scroll_page

PRESETS = {
    "smoke": ("Smoke test: check the whole pipeline in about two minutes", dict(worlds=16, workers=2, threads=2, updates=5, every=2)),
    "diagnostic": ("Diagnostic: about 1 million decisions (about 10 minutes)", dict(worlds=64, workers=8, threads=2, updates=250, every=25)),
    "extended": ("Extended diagnostic: about 2 million decisions (about 20 minutes)", dict(worlds=64, workers=8, threads=2, updates=500, every=50)),
    "custom": ("Custom budget", {}),
}


class ConfirmDialog(QtWidgets.QDialog):
    def __init__(self, plan: launch.LaunchPlan, warnings: list[launch.Check], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start training?")
        self.setMinimumWidth(560)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(make_label("<h3>Start this run?</h3>"))
        rows = [("Experiment", plan.experiment.name), ("Run folder", str(plan.run_dir)),
                ("Starting point", f"Continue from update {plan.resume.update} of {plan.resume_run.run_id if plan.resume_run else plan.resume.path.parent.name}"
                 if plan.resume else "New run, untrained policy"),
                ("Simulation", f"{plan.worlds} worlds × {plan.workers} workers × {plan.physx_threads} PhysX threads, PPO on {plan.device.upper()}"),
                ("Seed", str(plan.seed)),
                ("Budget", f"{plan.updates} updates = {human_count(plan.total_decisions)} decisions "
                           f"(about {human_duration(plan.estimated_seconds())} at {RECOMMENDED['throughput']:,.0f} decisions/s)"),
                ("Checkpoints", f"latest after every update" + (f", numbered every {plan.checkpoint_every} updates" if plan.checkpoint_every else "")
                 + (", plus the untrained initial policy" if plan.save_initial and not plan.resume else "")),
                ("Memory / disk", f"about {human_bytes(plan.estimated_memory_bytes())} RAM, {human_bytes(plan.estimated_disk_bytes())} disk")]
        grid = QtWidgets.QGridLayout()
        for i, (k, v) in enumerate(rows):
            grid.addWidget(make_label(f"<span style='color:gray'>{k}</span>", wrap=False), i, 0, QtCore.Qt.AlignTop)
            grid.addWidget(make_label(v, selectable=True), i, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        if warnings:
            layout.addWidget(Banner("warn", f"{len(warnings)} warning(s) you have seen", "\n".join(f"• {w.title}" for w in warnings)))
        layout.addWidget(make_label("The run keeps going if you close the Studio. You can stop it gracefully from the Training page; the checkpoint of the last finished update is kept.", "hint"))
        buttons = QtWidgets.QDialogButtonBox()
        start = buttons.addButton("Start training", QtWidgets.QDialogButtonBox.AcceptRole)
        start.setObjectName("primary")
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ExperimentsPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self._loading = False
        self._resume_choices: list[tuple[CheckpointInfo, RunRecord | None]] = []
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # ---- left: list
        left = QtWidgets.QWidget()
        left.setFixedWidth(300)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(18, 22, 8, 18)
        left_layout.addWidget(make_label("Experiments", "pageTitle", wrap=False))
        left_layout.addWidget(make_label("Pick one, review it, configure the run.", "hint"))
        self.list = QtWidgets.QListWidget()
        self.list.setAlternatingRowColors(False)
        self.list.setWordWrap(True)
        self.list.setSpacing(3)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.currentRowChanged.connect(self._selected)
        left_layout.addWidget(self.list, 1)
        outer.addWidget(left)
        # ---- right: detail
        content, self.layout_ = page_container()
        outer.addWidget(scroll_page(content), 1)
        self.header = PageHeader("", "")
        self.start_top = QtWidgets.QPushButton("Start training")
        self.start_top.setObjectName("primary")
        self.start_top.clicked.connect(self._start)
        self.header.actions.addWidget(self.start_top)
        self.layout_.addWidget(self.header)
        self.chips = make_label("", "hint")
        self.layout_.addWidget(self.chips)
        self.error_banner = Banner("bad", "", "")
        self.error_banner.hide()
        self.layout_.addWidget(self.error_banner)
        self.limit_banner = Banner("warn", "Known limitation of XY1 as defined: agents can fall out of the world",
                                   "The floor of a world can be destroyed, and nothing ends an episode when the agent falls out of the world. In real runs "
                                   "about a fifth of the agents are gone after roughly 20 updates and most after 100; they are lost to training until the 100,000-decision ceiling. "
                                   "The Studio measures this (chart “Agents that left the world”) but does not change the experiment. See the learning-readiness report.")
        self.layout_.addWidget(self.limit_banner)
        self.about = Card("About this experiment")
        self.about_text = make_label("")
        self.about.add(self.about_text)
        self.layout_.addWidget(self.about)
        self.fixed = Card("Scientific contract: fixed by the experiment", "The Studio shows these and never changes them. To change an experiment, edit its file and keep it under version control.",
                          help_callback=ctx.show_help)
        self.fixed_grid = QtWidgets.QGridLayout()
        self.fixed_grid.setHorizontalSpacing(14)
        self.fixed_grid.setVerticalSpacing(5)
        self.fixed.add(self.fixed_grid)
        self.layout_.addWidget(self.fixed)
        self._build_workload()
        self.validation = Card("Validation", "Re-checked whenever you change something.")
        self.verdict = Banner("info", "", "")
        self.validation.add(self.verdict)
        self.checks_box = QtWidgets.QVBoxLayout()
        self.checks_box.setSpacing(3)
        self.validation.add(self.checks_box)
        row = QtWidgets.QHBoxLayout()
        self.command_button = QtWidgets.QPushButton("Show launch command")
        self.command_button.clicked.connect(self._show_command)
        row.addWidget(self.command_button)
        row.addStretch(1)
        self.start_bottom = QtWidgets.QPushButton("Start training")
        self.start_bottom.setObjectName("primary")
        self.start_bottom.clicked.connect(self._start)
        row.addWidget(self.start_bottom)
        self.validation.add(row)
        self.layout_.addWidget(self.validation)
        self.layout_.addStretch(1)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self.revalidate)
        ctx.runsChanged.connect(self._runs_changed)
        self._fill_list()

    # ------------------------------------------------------------------------------------------ form
    def _build_workload(self):
        ctx = self.ctx
        self.workload = Card("Training workload: you choose", "Execution settings. They change speed and budget, never the science.", help_callback=ctx.show_help)
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.preset = QtWidgets.QComboBox()
        for key, (label, _v) in PRESETS.items():
            self.preset.addItem(label, key)
        self.preset.setCurrentIndex(1)
        self.worlds = QtWidgets.QSpinBox(); self.worlds.setRange(1, 256); self.worlds.setValue(RECOMMENDED["worlds"])
        self.workers = QtWidgets.QSpinBox(); self.workers.setRange(1, 16); self.workers.setValue(RECOMMENDED["workers"])
        self.threads = QtWidgets.QSpinBox(); self.threads.setRange(1, 8); self.threads.setValue(RECOMMENDED["physx_threads"])
        self.device = QtWidgets.QComboBox(); self.device.addItem("CUDA (GPU)", "cuda"); self.device.addItem("CPU (very slow)", "cpu")
        self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 2**31 - 1); self.seed.setValue(7)
        self.updates = QtWidgets.QSpinBox(); self.updates.setRange(1, 1_000_000); self.updates.setValue(250)
        self.every = QtWidgets.QSpinBox(); self.every.setRange(0, 100000); self.every.setValue(25); self.every.setSpecialValueText("off")
        self.initial = QtWidgets.QCheckBox("Also save the untrained initial policy (baseline for evaluation)"); self.initial.setChecked(True)
        self.run_name = QtWidgets.QLineEdit()
        self.start_from = QtWidgets.QComboBox()
        self.start_from.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.start_from.setMinimumContentsLength(30)
        rows = [("Budget", self.preset, None), ("Starting point", self.start_from, "resume"), ("Worlds", self.worlds, "worlds"), ("Workers", self.workers, "worlds"),
                ("PhysX threads per worker", self.threads, "worlds"), ("PPO device", self.device, None), ("Random seed", self.seed, None),
                ("Updates", self.updates, "update"), ("Checkpoint every", self.every, "checkpoint"), ("", self.initial, None), ("Run name", self.run_name, None)]
        for i, (label, widget, help_key) in enumerate(rows):
            form.addWidget(make_label(label, wrap=False), i, 0, QtCore.Qt.AlignVCenter)
            form.addWidget(widget, i, 1)
            if help_key:
                form.addWidget(InfoButton(help_key, ctx.show_help), i, 2)
        form.setColumnStretch(1, 1)
        self.workload.add(form)
        self.summary = Banner("info", "", "")
        self.workload.add(self.summary)
        self.layout_.addWidget(self.workload)
        for w in (self.worlds, self.workers, self.threads, self.seed, self.updates, self.every):
            w.valueChanged.connect(self._field_changed)
        self.device.currentIndexChanged.connect(self._field_changed)
        self.initial.toggled.connect(self._field_changed)
        self.run_name.textEdited.connect(self._name_edited)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.start_from.currentIndexChanged.connect(self._start_from_changed)
        self._name_custom = False

    def _preset_changed(self):
        key = self.preset.currentData()
        values = PRESETS[key][1]
        if not values:
            return
        self._loading = True
        self.worlds.setValue(values["worlds"]); self.workers.setValue(values["workers"]); self.threads.setValue(values["threads"])
        self.updates.setValue(values["updates"]); self.every.setValue(values["every"])
        self._loading = False
        self._field_changed(from_preset=True)

    def _field_changed(self, *_args, from_preset=False):
        if self._loading:
            return
        if not from_preset and self.preset.currentData() != "custom":
            matches = [k for k, (_l, v) in PRESETS.items() if v and (v["worlds"], v["workers"], v["threads"], v["updates"], v["every"]) ==
                       (self.worlds.value(), self.workers.value(), self.threads.value(), self.updates.value(), self.every.value())]
            self._loading = True
            self.preset.setCurrentIndex(self.preset.findData(matches[0] if matches else "custom"))
            self._loading = False
        if not self._name_custom:
            self._regen_name()
        self._debounce.start()

    def _name_edited(self, _text):
        self._name_custom = True
        self._debounce.start()

    def _regen_name(self):
        exp = self.ctx.selected_experiment
        if exp:
            self._loading = True
            self.run_name.setText(launch.default_run_name(exp, self.seed.value()))
            self._loading = False

    # ------------------------------------------------------------------------------------------ experiment list
    def _fill_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for exp in self.ctx.experiments:
            tag = "  ★ recommended" if exp.is_mini else ""
            item = QtWidgets.QListWidgetItem(f"{exp.name}{tag}")
            item.setToolTip(str(exp.path))
            self.list.addItem(item)
        self.list.blockSignals(False)
        sel = self.ctx.selected_experiment
        index = next((i for i, e in enumerate(self.ctx.experiments) if sel and e.path == sel.path), 0)
        self.list.setCurrentRow(index)
        self._selected(index)

    def _selected(self, row: int):
        if row < 0 or row >= len(self.ctx.experiments):
            return
        exp = self.ctx.experiments[row]
        self.ctx.select_experiment(exp)
        self._show_experiment(exp)

    def _show_experiment(self, exp: ExperimentInfo):
        self.header.title.setText(exp.name)
        self.header.subtitle.setText(str(exp.path.relative_to(launch.env.ROOT)) if exp.path.is_relative_to(launch.env.ROOT) else str(exp.path))
        chips = []
        if exp.is_mini:
            chips.append("★ validated layout: 64 worlds × 8 workers")
        if exp.error:
            chips.append("✖ file does not load")
        elif exp.mismatches:
            chips.append("⚠ differs from the intended XY1 contract")
        self.chips.setText("   ·   ".join(chips))
        if exp.error:
            self.error_banner.set("bad", "This experiment file does not load", exp.error)
            self.error_banner.show()
            self.limit_banner.hide()
            for w in (self.about, self.fixed, self.workload):
                w.setVisible(False)
        else:
            self.error_banner.hide()
            for w in (self.about, self.fixed, self.workload):
                w.setVisible(True)
            self.limit_banner.setVisible(exp.is_xy1)
            self.about_text.setText(f"{exp.description}<br><br><b>Hypothesis.</b> {exp.hypothesis}<br><br><span class='muted'>{exp.notes}</span>")
            self._fill_fixed(exp)
        self._regen_name()
        self._refresh_start_from()
        self._debounce.start()

    def _fill_fixed(self, exp: ExperimentInfo):
        clear_layout(self.fixed_grid)
        s = exp.summary
        heads = ", ".join(f"{a['name']} ({a['size']})" for a in s["actions"])
        spatial = next((f for f in s["observations"] if len(f["shape"]) == 3), None)
        scalar = next((f for f in s["observations"] if len(f["shape"]) == 1), None)
        ppo = s["ppo"]
        reward = "; ".join(f"{t['name']} × {t['weight']}" for t in s["reward_terms"]) or "see experiment file"
        env = s.get("environment") or {}
        rows = [("World", "128 × 128 tiles, independently destructible terrain per world (different seed per world)" if exp.is_mini else "Canonical world", "environment"),
                ("Observation", f"{'×'.join(map(str, spatial['shape']))} terrain/fluid channels around the agent + {scalar['shape'][0]} goal offsets (tiles)" if spatial and scalar else "–", None),
                ("Actions", f"{len(s['actions'])} heads: {heads}", None),
                ("Decision cadence", f"{s['physics_ticks_per_decision']} physics ticks of 20 ms per decision (0.2 s)", "decision"),
                ("Reward", reward, "reward"), ("Episode ends", "arrival within one tile of the goal, or the decision ceiling (100,000 in training)", "arrival"),
                ("Network", "convolutional policy + value network (parameter count printed at launch)", None),
                ("PPO", f"rollout {ppo['rollout_steps']} decisions per world, minibatch {ppo['minibatch_size']}, {ppo['epochs']} epochs, learning rate {ppo['learning_rate']}, "
                        f"γ {ppo['gamma']}, λ {ppo['gae_lambda']}, clip {ppo['clip_range']}, entropy coefficient {ppo['entropy_coefficient']}, target KL {ppo['target_kl']}", "update")]
        for i, (k, v, help_key) in enumerate(rows):
            self.fixed_grid.addWidget(make_label(f"🔒 {k}", wrap=False), i, 0, QtCore.Qt.AlignTop)
            self.fixed_grid.addWidget(make_label(v), i, 1)
            if help_key:
                self.fixed_grid.addWidget(InfoButton(help_key, self.ctx.show_help), i, 2, QtCore.Qt.AlignTop)
        self.fixed_grid.setColumnStretch(1, 1)

    # ------------------------------------------------------------------------------------------ starting point (resume)
    def _runs_changed(self):
        self._refresh_start_from(keep=True)

    def _refresh_start_from(self, keep: bool = False):
        exp = self.ctx.selected_experiment
        previous = self.start_from.currentData() if keep else None
        self.start_from.blockSignals(True)
        self.start_from.clear()
        self.start_from.addItem("New run: untrained policy", None)
        self._resume_choices = []
        if exp and exp.fingerprint:
            for run in self.ctx.runs:
                if run.latest and run.status not in ("starting", "running", "stopping") and run.latest.fingerprint in (None, exp.fingerprint):
                    self._resume_choices.append((run.latest, run))
                    self.start_from.addItem(f"Continue “{run.run_id}” from update {run.latest.update if run.latest.update is not None else '?'}", str(run.latest.path))
        self.start_from.addItem("Browse for a checkpoint file…", "browse")
        index = self.start_from.findData(previous) if previous else 0
        self.start_from.setCurrentIndex(max(0, index))
        self.start_from.blockSignals(False)

    def _start_from_changed(self):
        data = self.start_from.currentData()
        if data == "browse":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a checkpoint", str(self.ctx.run_root), "Checkpoints (*.pt)")
            if path:
                from ..runs import checkpoint_info
                info = checkpoint_info(Path(path))
                self._resume_choices.append((info, None))
                self.start_from.blockSignals(True)
                self.start_from.insertItem(self.start_from.count() - 1, f"File: {Path(path).parent.name}/{Path(path).name}", path)
                self.start_from.setCurrentIndex(self.start_from.count() - 2)
                self.start_from.blockSignals(False)
            else:
                self.start_from.blockSignals(True)
                self.start_from.setCurrentIndex(0)
                self.start_from.blockSignals(False)
        self._apply_resume_layout()
        self._debounce.start()

    def _selected_resume(self) -> tuple[CheckpointInfo | None, RunRecord | None]:
        data = self.start_from.currentData()
        if not data or data == "browse":
            return None, None
        for info, run in self._resume_choices:
            if str(info.path) == data:
                return info, run
        return None, None

    def _apply_resume_layout(self):
        info, run = self._selected_resume()
        locked = info is not None and info.world_partition is not None
        for w in (self.worlds, self.workers):
            w.setEnabled(not locked)
        self.initial.setEnabled(info is None)
        if locked:
            self._loading = True
            self.worlds.setValue(sum(info.world_partition))
            self.workers.setValue(len(info.world_partition))
            self._loading = False
        if info is not None:
            self.worlds.setToolTip("Fixed: must match the world layout stored in the checkpoint")

    def on_show(self, payload=None):
        self.ctx.reload_experiments()
        if isinstance(payload, dict) and payload.get("resume"):
            self._refresh_start_from()
            index = self.start_from.findData(payload["resume"])
            if index < 0:
                from ..runs import checkpoint_info
                info = checkpoint_info(Path(payload["resume"]))
                run = self.ctx.run_by_path(payload.get("run")) if payload.get("run") else None
                self._resume_choices.append((info, run))
                self.start_from.insertItem(1, f"Continue “{Path(payload['resume']).parent.name}” from update {info.update}", payload["resume"])
                index = 1
            self.start_from.setCurrentIndex(index)
        self._debounce.start()

    # ------------------------------------------------------------------------------------------ plan / validation
    def current_plan(self) -> launch.LaunchPlan | None:
        exp = self.ctx.selected_experiment
        if exp is None:
            return None
        info, run = self._selected_resume()
        return launch.LaunchPlan(experiment=exp, run_root=self.ctx.run_root, run_name=self.run_name.text().strip() or launch.default_run_name(exp, self.seed.value()),
                                 worlds=self.worlds.value(), workers=self.workers.value(), physx_threads=self.threads.value(), device=self.device.currentData(),
                                 seed=self.seed.value(), updates=self.updates.value(), checkpoint_every=self.every.value(), save_initial=self.initial.isChecked(),
                                 resume=info, resume_run=run)

    def revalidate(self):
        plan = self.current_plan()
        if plan is None or plan.experiment.error:
            self.start_top.setEnabled(False)
            self.start_bottom.setEnabled(False)
            return
        owned = {self.ctx.session.pid} if self.ctx.session else None
        checks = launch.validate(plan, self.ctx.install, owned_pids=owned)
        self._checks = checks
        palette = self.ctx.palette
        clear_layout(self.checks_box)
        glyph = {launch.OK: ("✔", palette.good), launch.WARN: ("⚠", palette.warn), launch.ERROR: ("✖", palette.bad)}
        for c in sorted(checks, key=lambda c: {launch.ERROR: 0, launch.WARN: 1, launch.OK: 2}[c.level]):
            g, color = glyph[c.level]
            label = make_label(f"<span style='color:{color}; font-weight:700'>{g}</span> <b>{c.title}</b>" + (f"<br><span class='muted'>{c.detail}</span>" if c.detail else ""))
            self.checks_box.addWidget(label)
        errors = launch.blocking(checks)
        warns = [c for c in checks if c.level == launch.WARN]
        if errors:
            self.verdict.set("bad", f"{len(errors)} problem(s) must be fixed before training can start", "; ".join(c.title for c in errors))
        elif warns:
            self.verdict.set("warn", "Ready to start, with warnings", f"{len(warns)} warning(s) below do not block the launch; read them first.")
        else:
            self.verdict.set("good", "Ready to start", "Every check passed.")
        reason = ("Fix: " + errors[0].title) if errors else ""
        for b in (self.start_top, self.start_bottom):
            b.setEnabled(not errors)
            b.setToolTip(reason)
        self.summary.set("info", f"Each update collects {plan.rollout_steps} decisions in each of {plan.worlds} worlds = {plan.decisions_per_update:,} decisions. "
                                 f"{plan.updates} updates = {human_count(plan.total_decisions)} decisions, about {human_duration(plan.estimated_seconds())} at "
                                 f"{RECOMMENDED['throughput']:,.0f} decisions/s.",
                         f"Estimated {human_bytes(plan.estimated_memory_bytes())} RAM and {human_bytes(plan.estimated_disk_bytes())} of disk. Checkpoints → {plan.run_dir}")

    def _show_command(self):
        plan = self.current_plan()
        if plan is None:
            return
        text = launch.command_text(launch.build_command(plan, self.ctx.install))
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Launch command")
        box.setText("This is exactly what the Studio runs (the Isaac environment variables are set up automatically):")
        box.setDetailedText(text)
        box.setInformativeText(text)
        box.exec()

    def _start(self):
        plan = self.current_plan()
        if plan is None:
            return
        self.revalidate()
        errors = launch.blocking(self._checks)
        if errors:
            QtWidgets.QMessageBox.warning(self, "Cannot start", "\n".join(c.title + ": " + c.detail for c in errors))
            return
        warns = [c for c in self._checks if c.level == launch.WARN]
        if ConfirmDialog(plan, warns, self).exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            self.ctx.start_training(plan)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not start", str(exc))
            return
        self._name_custom = False
        self.ctx.navigate.emit("training", plan.run_dir)
