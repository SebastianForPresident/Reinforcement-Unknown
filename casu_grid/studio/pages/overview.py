"""Overview: what is the state of the project and what should I do next?"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import analysis, environment as env, launch, theme
from ..runs import MetricsStore, RunRecord, human_bytes, human_count, human_duration
from ..widgets import (Banner, Card, FlowRow, PageHeader, StatTile, StatusBadge, clear_layout, make_label, page_container, scroll_page)


class RunCard(QtWidgets.QFrame):
    clicked = QtCore.Signal(object)

    def __init__(self, run: RunRecord, palette: theme.Palette, signal_text: str):
        super().__init__()
        self.run = run
        self.setObjectName("card")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        badge = StatusBadge()
        badge.set_palette(palette)
        badge.set_status(run.status)
        badge.setFixedWidth(120)
        layout.addWidget(badge)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(1)
        kind = "  (engineering smoke test)" if run.is_engineering_smoke else ""
        text.addWidget(make_label(f"<b>{run.run_id}</b>{kind}", wrap=False))
        layout.addLayout(text, 1)
        facts = f"{run.updates_done} updates · {human_count(run.decisions)} decisions"
        if run.num_envs:
            facts += f" · {run.num_envs} worlds" + (f" × {run.workers} workers" if run.workers else "")
        text.addWidget(make_label(facts, "hint", wrap=False))
        if signal_text:
            text.addWidget(make_label(signal_text, "hint", wrap=False))

    def mouseReleaseEvent(self, e):
        self.clicked.emit(self.run)


class OverviewPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content, self.layout_ = page_container()
        outer.addWidget(scroll_page(content))
        self.header = PageHeader("Overview", "")
        self.layout_.addWidget(self.header)
        self.hero = QtWidgets.QFrame()
        self.hero.setObjectName("heroCard")
        hero_layout = QtWidgets.QVBoxLayout(self.hero)
        hero_layout.setContentsMargins(20, 16, 20, 16)
        self.hero_kicker = make_label("WHAT TO DO NEXT", "hint", wrap=False)
        self.hero_title = make_label("", "pageTitle")
        self.hero_title.setStyleSheet("font-size:18px;")
        self.hero_text = make_label("")
        self.hero_actions = QtWidgets.QHBoxLayout()
        self.hero_actions.addStretch(1)
        for w in (self.hero_kicker, self.hero_title, self.hero_text):
            hero_layout.addWidget(w)
        hero_layout.addLayout(self.hero_actions)
        self.layout_.addWidget(self.hero)
        self.layout_.addWidget(make_label("<b>System readiness</b>"))
        self.ready_row = FlowRow(200)
        self.tiles = {name: StatTile(label) for name, label in
                      (("isaac", "Simulator"), ("gpu", "GPU"), ("memory", "Memory"), ("source", "Source revision"), ("legacy", "Legacy spawn mode"))}
        for t in self.tiles.values():
            self.ready_row.add(t)
        self.layout_.addWidget(self.ready_row)
        self.issue_box = QtWidgets.QVBoxLayout()
        self.layout_.addLayout(self.issue_box)
        self.exp_card = Card("Selected experiment")
        self.exp_text = make_label("")
        self.exp_card.add(self.exp_text)
        row = QtWidgets.QHBoxLayout()
        review = QtWidgets.QPushButton("Review configuration")
        review.clicked.connect(lambda: self.ctx.navigate.emit("experiments", None))
        row.addWidget(review)
        row.addStretch(1)
        self.exp_card.add(row)
        self.layout_.addWidget(self.exp_card)
        self.layout_.addWidget(make_label("<b>Latest runs</b>"))
        self.runs_box = QtWidgets.QVBoxLayout()
        self.layout_.addLayout(self.runs_box)
        self.layout_.addStretch(1)
        ctx.runsChanged.connect(self.refresh)
        ctx.themeChanged.connect(self.refresh)
        self._cache_time = 0.0
        self._cached = None
        self.refresh()

    def on_show(self, payload=None):
        self.refresh(force=True)

    # --------------------------------------------------------------------------------------------- content
    def _machine(self, force: bool):
        import time
        if force or self._cached is None or time.time() - self._cache_time > 10:
            self._cached = (env.source_info(), env.gpu_info(), env.memory_info(), env.inherited_overrides(), env.stray_simulator_processes())
            self._cache_time = time.time()
        return self._cached

    def _signal_text(self, run: RunRecord) -> str:
        if not run.has_metrics:
            return ""
        key = (str(run.path), run.updates_done)
        cache = self.__dict__.setdefault("_sig_cache", {})
        if key in cache:
            return cache[key]
        cache[key] = text = self._signal_text_uncached(run)
        return text

    def _signal_text_uncached(self, run: RunRecord) -> str:
        store = MetricsStore(run.path / "metrics.jsonl")
        store.refresh()
        arrivals = analysis.mean_of_last(store, "nav/arrivals_per_1k_decisions", 10)
        if arrivals is None:
            return "navigation statistics not recorded in this run"
        return f"goals reached: {arrivals:.2f} per 1,000 decisions (last 10 updates)"

    def refresh(self, force: bool = False):
        ctx = self.ctx
        exp = ctx.selected_experiment
        self.header.subtitle.setText(f"{exp.name}" if exp else "No experiment found")
        source, gpu, mem, overrides, strays = self._machine(force)
        palette = ctx.palette
        ok_color, bad_color, warn_color = palette.good, palette.bad, palette.warn
        self.tiles["isaac"].set("✔ ready" if ctx.install.ok else "✖ not found", str(ctx.install.sim_root) if ctx.install.sim_root else "; ".join(ctx.install.problems),
                                ok_color if ctx.install.ok else bad_color)
        if gpu:
            free = gpu["total_mib"] - gpu["used_mib"]
            self.tiles["gpu"].set(f"✔ {free / 1024:.1f} GiB free", gpu["name"], ok_color if free > 4096 else warn_color)
        else:
            self.tiles["gpu"].set("✖ none visible", "CUDA training needs an NVIDIA GPU", bad_color)
        avail = mem["available"]
        self.tiles["memory"].set(f"{human_bytes(avail)} free", f"of {human_bytes(mem['total'])}; swap in use {human_bytes(mem['swap_used'])}",
                                 ok_color if avail > 20 * 2**30 else warn_color)
        if source.revision:
            self.tiles["source"].set(f"{source.short} · {source.branch}",
                                     ("includes Optimization IV corrections ✔" if source.has_all_fixes else "MISSING corrections ✖")
                                     + (f"; {source.dirty_files} uncommitted file(s)" if source.dirty_files else ""),
                                     ok_color if source.has_all_fixes else bad_color)
        else:
            self.tiles["source"].set("unknown", "not a git checkout", bad_color)
        legacy = overrides.get("CASU_LEGACY_MINI_SPAWN") == "1"
        self.tiles["legacy"].set("✖ ON (refused)" if legacy else "✔ off", "historical reproduction only" if not legacy else "unset it to train", bad_color if legacy else ok_color)
        clear_layout(self.issue_box)
        if strays:
            banner = Banner("warn", "Simulator processes are already running",
                            f"{len(strays)} process(es) from another run or a crash are using the CPU (e.g. pid {strays[0][0]}). Starting another run is blocked until they end.")
            self.issue_box.addWidget(banner)
        # experiment card
        if exp and not exp.error:
            s = exp.summary
            self.exp_text.setText(
                f"<b>{exp.name}</b><br>{exp.description}<br><span class='muted'>Hypothesis: {exp.hypothesis}</span><br>"
                f"{len(s['actions'])} action heads · {s['physics_ticks_per_decision']} physics ticks per decision · PPO rollout {s['ppo']['rollout_steps']}, "
                f"minibatch {s['ppo']['minibatch_size']}, {s['ppo']['epochs']} epochs")
        elif exp:
            self.exp_text.setText(f"<b>{exp.name}</b><br>The experiment file does not load: {exp.error}")
        # runs
        clear_layout(self.runs_box)
        runs = [r for r in ctx.runs if not exp or r.config_path is None or Path(r.config_path).name == exp.path.name or True][:5]
        if not runs:
            self.runs_box.addWidget(make_label("No runs yet. They will appear here as soon as a training run is started.", "hint"))
        for run in runs:
            card = RunCard(run, palette, self._signal_text(run) if run.status != "unknown" and len(runs) <= 5 else "")
            card.clicked.connect(lambda r: ctx.navigate.emit("runs", r.path))
            self.runs_box.addWidget(card)
        self._hero(exp, runs, source, strays, legacy)

    def _clear_actions(self):
        while self.hero_actions.count() > 1:
            widget = self.hero_actions.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _hero_button(self, text: str, page: str, payload=None, primary=False):
        b = QtWidgets.QPushButton(text)
        if primary:
            b.setObjectName("primary")
        b.clicked.connect(lambda: self.ctx.navigate.emit(page, payload))
        self.hero_actions.insertWidget(self.hero_actions.count() - 1, b)

    def _hero(self, exp, runs, source, strays, legacy):
        ctx = self.ctx
        self._clear_actions()
        research = [r for r in runs if not r.is_engineering_smoke]
        active = ctx.active_run()
        failed = runs[0] if runs and runs[0].status == "failed" else None
        if not ctx.install.ok:
            self.hero_title.setText("Isaac Sim was not found")
            self.hero_text.setText("The simulator runs in Isaac Sim's own Python. " + " ".join(ctx.install.problems))
            self._hero_button("Open help", "help", primary=True)
        elif legacy:
            self.hero_title.setText("Legacy spawn mode is switched on")
            self.hero_text.setText("CASU_LEGACY_MINI_SPAWN=1 is set in the environment that started the Studio. Training and evaluation are refused while it is set: "
                                   "it recreates the mis-placed worlds of earlier benchmarks. Unset it and restart the Studio.")
        elif active is not None:
            self.hero_title.setText(f"Training is {active.status}: update {active.updates_done}")
            store = MetricsStore(active.path / "metrics.jsonl")
            store.refresh()
            sig = analysis.learning_signal(store)
            self.hero_text.setText(f"{human_count(active.decisions)} decisions so far. {sig.headline}.")
            self._hero_button("Open training monitor", "training", primary=True)
            self._hero_button("Open metrics", "metrics", active.path)
        elif failed is not None:
            headline, _ = launch.explain_failure(failed)
            self.hero_title.setText("The last run failed")
            self.hero_text.setText(headline + (f" A checkpoint from update {failed.latest.update} is still available." if failed.latest else " No checkpoint was saved."))
            self._hero_button("See what happened", "training", failed.path, primary=True)
            if failed.latest:
                self._hero_button("Resume from checkpoint", "experiments", {"resume": str(failed.latest.path), "run": str(failed.path)})
        elif not research:
            self.hero_title.setText("No learning run exists yet")
            self.hero_text.setText(
                "XY1 has not yet been trained with the corrected 64-world setup, so there is no evidence of learning to show. The speed measurements from the "
                "optimisation work say how fast it runs, not whether the agent learns. Review the configuration and start with a short smoke test, "
                "or go straight to the 1-million-decision diagnostic run.")
            self._hero_button("Review configuration and start", "experiments", primary=True)
            self._hero_button("How does this work?", "help")
        else:
            latest = research[0]
            unevaluated = not latest.evaluations
            self.hero_title.setText(f"Latest run “{latest.run_id}” {theme.STATUS[latest.status][1].lower()} after {latest.updates_done} updates")
            if unevaluated and latest.latest:
                self.hero_text.setText("Training curves show how the agent behaved while exploring on the training terrains. Evaluate its checkpoint on held-out terrains, "
                                       "next to the untrained baseline, to see whether it really learned.")
                self._hero_button("Evaluate this checkpoint", "evaluation", {"run": str(latest.path)}, primary=True)
            else:
                self.hero_text.setText("Inspect the curves, compare runs, or continue training from the latest checkpoint.")
                self._hero_button("Open metrics", "metrics", latest.path, primary=True)
            self._hero_button("Runs & checkpoints", "runs", latest.path)
            if latest.latest and latest.status in ("interrupted", "completed"):
                self._hero_button("Continue training", "experiments", {"resume": str(latest.latest.path), "run": str(latest.path)})
