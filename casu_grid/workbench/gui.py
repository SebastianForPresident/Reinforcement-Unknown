"""Legacy (V1) PySide6 research studio backed by the shared workbench CLI/services.

Superseded by Research Studio V2 (``python3 -m casu_grid.studio``), which has the guided XY1 workflow, interactive charts, supervised
launch/stop/resume and held-out evaluation. This module is kept working for compatibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict

from PySide6 import QtCore, QtGui, QtWidgets

from . import components, lab

ROOT = lab.ROOT
CONFIGS = ROOT / "casu_grid" / "workbench" / "experiments"
COMPONENTS = ROOT / "casu_grid" / "workbench" / "components"


class ContractDialog(QtWidgets.QDialog):
    """Typed, bounded editor for the supported scalar model and PPO contracts."""
    def __init__(self, model, ppo, parent=None):
        super().__init__(parent);self.setWindowTitle("Model and PPO parameters");self.resize(700,650)
        layout=QtWidgets.QVBoxLayout(self);tabs=QtWidgets.QTabWidget();layout.addWidget(tabs)
        self.controls={}
        for section,values in (("MODEL",asdict(model)),("PPO",asdict(ppo))):
            page=QtWidgets.QWidget();form=QtWidgets.QFormLayout(page);tabs.addTab(page,section)
            for key,value in values.items():
                if isinstance(value,bool):control=QtWidgets.QCheckBox();control.setChecked(value);kind="bool"
                elif isinstance(value,int) and not isinstance(value,bool):control=QtWidgets.QSpinBox();control.setRange(1,1_000_000);control.setValue(value);kind="int"
                elif isinstance(value,float):control=QtWidgets.QDoubleSpinBox();control.setRange(-1e9,1e9);control.setDecimals(9);control.setValue(value);kind="float"
                elif value is None:control=QtWidgets.QLineEdit("");control.setPlaceholderText("None or a finite number");kind="optional_float"
                elif isinstance(value,tuple):control=QtWidgets.QLineEdit(json.dumps(value));kind="tuple"
                else:continue
                control.setToolTip(key.replace("_"," ").capitalize())
                form.addRow(key.replace("_"," ").title(),control);self.controls[(section,key)]=(control,kind)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
    def values(self):
        sections={"MODEL":{},"PPO":{}}
        for (section,key),(control,kind) in self.controls.items():
            if kind=="bool":value=control.isChecked()
            elif kind=="int":value=control.value()
            elif kind=="float":value=control.value()
            elif kind=="optional_float":
                raw=control.text().strip()
                try:value=None if not raw or raw.lower()=="none" else float(raw)
                except ValueError as exc:raise ValueError(f"{key} must be None or a finite number") from exc
                if value is not None and not __import__("math").isfinite(value):raise ValueError(f"{key} must be finite")
            elif kind=="tuple":
                try:value=tuple(json.loads(control.text()))
                except Exception as exc:raise ValueError(f"{key} must be a JSON array") from exc
                if any(not isinstance(x,(int,float)) or isinstance(x,bool) for x in value):raise ValueError(f"{key} must contain numeric values")
            sections[section][key]=value
        return sections


class CommandWorker(QtCore.QThread):
    done = QtCore.Signal(int, str, str)
    def __init__(self, args: list[str], parent=None, executable: str | None = None, environment=None):
        super().__init__(parent); self.args = args; self.executable=executable or sys.executable; self.environment=environment
    def run(self):
        result = subprocess.run([self.executable, "-m", "casu_grid.workbench.lab", *self.args],
                                cwd=ROOT, env=self.environment, capture_output=True, text=True)
        self.done.emit(result.returncode, result.stdout, result.stderr)


class RuntimeInfoWorker(QtCore.QThread):
    done = QtCore.Signal(str)
    def __init__(self, executable, environment, parent=None):
        super().__init__(parent);self.executable=executable;self.environment=environment
    def run(self):
        code="""import importlib.metadata as m,json,platform,sys
out={'python':sys.version,'executable':sys.executable,'platform':platform.platform()}
for package,key in [('torch','pytorch_version'),('isaacsim','isaac_sim_package'),('isaaclab','isaac_lab_package')]:
 try:out[key]=m.version(package)
 except m.PackageNotFoundError:out[key]='not installed'
try:
 import torch
 out['gpu_available']=torch.cuda.is_available()
 if out['gpu_available']:
  out['gpu_name']=torch.cuda.get_device_name(0);out['gpu_total_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
  free,total=torch.cuda.mem_get_info();out['gpu_free_memory_bytes']=free;out['gpu_runtime_total_memory_bytes']=total
except Exception as e:out['gpu_probe_error']=str(e)
print(json.dumps(out,indent=2))"""
        try:
            p=subprocess.run([self.executable,"-c",code],env=self.environment,capture_output=True,text=True,timeout=30)
            self.done.emit(p.stdout+("\n"+p.stderr if p.stderr else "") if p.returncode==0 else f"Runtime query failed ({p.returncode}):\n{p.stderr}")
        except Exception as exc:self.done.emit(f"Runtime query failed: {exc}")


class Studio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reinforcement-Unknown Research Studio")
        self.resize(1280, 820)
        self.settings = QtCore.QSettings("ReinforcementUnknown", "ResearchStudio")
        if self.settings.value("isaac_sim"):
            os.environ.setdefault("ISAAC_SIM",str(self.settings.value("isaac_sim")))
        if self.settings.value("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))
        self.run_root=Path(os.environ.get("CASU_STUDIO_RUN_ROOT",self.settings.value("run_root",str(ROOT/"experiments")))).expanduser().resolve()
        root = QtWidgets.QWidget(); self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        self.nav = QtWidgets.QListWidget(); self.nav.setFixedWidth(185)
        for item in ("Laboratory", "Experiments", "Experiment editor", "Components", "Training", "Runs & metrics", "Compare", "Replay", "Environment", "Help"):
            self.nav.addItem(item)
        layout.addWidget(self.nav)
        self.pages = QtWidgets.QStackedWidget(); layout.addWidget(self.pages, 1)
        self._make_home(); self._make_experiments(); self._make_editor(); self._make_components()
        self._make_training(); self._make_runs(); self._make_compare(); self._make_replay()
        self._make_environment(); self._make_help()
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)
        self.statusBar().showMessage("Local workspace · no Isaac process started")
        self.refresh_all()

    def _page(self, title: str, intro: str = ""):
        page = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(page)
        heading = QtWidgets.QLabel(title); heading.setStyleSheet("font-size:22px;font-weight:600")
        v.addWidget(heading)
        if intro: v.addWidget(QtWidgets.QLabel(intro))
        self.pages.addWidget(page)
        return page, v

    def _button(self, label, fn):
        b = QtWidgets.QPushButton(label); b.clicked.connect(fn); return b

    def _log(self, parent_layout):
        box = QtWidgets.QPlainTextEdit(); box.setReadOnly(True); box.setMaximumBlockCount(2500)
        parent_layout.addWidget(box, 1); return box

    def _make_home(self):
        p, v = self._page("Research laboratory", "A local interface to the existing CasU Isaac workbench. Lightweight browsing does not start Isaac.")
        self.home_info = QtWidgets.QLabel(); self.home_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v.addWidget(self.home_info)
        row = QtWidgets.QHBoxLayout()
        for title, page in [("Create / open experiment", 1), ("Component library", 3), ("Inspect runs", 5), ("Environment status", 8), ("Documentation", 9)]:
            row.addWidget(self._button(title, lambda checked=False, n=page: self.nav.setCurrentRow(n)))
        v.addLayout(row); v.addWidget(QtWidgets.QLabel("Recent experiments and runs"))
        self.home_recent = QtWidgets.QPlainTextEdit(); self.home_recent.setReadOnly(True); v.addWidget(self.home_recent)

    def _make_experiments(self):
        p, v = self._page("Experiments", "Browse Python research contracts. Example defaults remain explicitly engineering placeholders.")
        row = QtWidgets.QHBoxLayout(); self.config_search = QtWidgets.QLineEdit(); self.config_search.setPlaceholderText("Search experiments")
        self.config_search.textChanged.connect(self.refresh_configs); row.addWidget(self.config_search)
        row.addWidget(self._button("New experiment…", self.create_experiment)); row.addWidget(self._button("Refresh", self.refresh_configs)); v.addLayout(row)
        self.config_table = QtWidgets.QTableWidget(0, 6); self.config_table.setHorizontalHeaderLabels(["Name", "Hypothesis", "Modified", "SHA-256", "Associated runs", "Configuration path"]); self.config_table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch); v.addWidget(self.config_table, 1)
        self.config_table.cellDoubleClicked.connect(lambda *_: self.open_selected_config())
        row = QtWidgets.QHBoxLayout(); row.addWidget(self._button("Open in editor", self.open_selected_config)); row.addWidget(self._button("Duplicate…", self.duplicate_config)); self.show_hidden=QtWidgets.QCheckBox("Show hidden experiments"); self.show_hidden.toggled.connect(self.refresh_configs); row.addWidget(self.show_hidden); row.addWidget(self._button("Hide / restore", self.toggle_hidden_config)); row.addStretch(); v.addLayout(row)

    def _make_editor(self):
        p, v = self._page("Experiment editor", "Python remains authoritative. The editor opens source directly; saving is explicit and checks for external changes.")
        row = QtWidgets.QHBoxLayout(); self.editor_path = QtWidgets.QLineEdit(); self.editor_path.setPlaceholderText("Select a Python experiment configuration")
        row.addWidget(self.editor_path, 1); row.addWidget(self._button("Browse…", self.browse_config)); row.addWidget(self._button("Load", self.load_editor)); v.addLayout(row)
        self.editor_hash = QtWidgets.QLabel("No configuration loaded"); v.addWidget(self.editor_hash)
        self.editor = QtWidgets.QPlainTextEdit(); self.editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap); v.addWidget(self.editor, 1)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self._button("Inspect contracts", self.inspect_editor)); row.addWidget(self._button("Configure model / PPO…", self.configure_model_ppo)); row.addWidget(self._button("Probe observations in Isaac…", self.probe_observations)); row.addWidget(self._button("Review diff", self.review_editor)); row.addWidget(self._button("Save safely", self.save_editor)); row.addStretch(); v.addLayout(row)
        self.editor_report = self._log(v)
        self._editor_original = None; self._editor_digest = None

    def _make_components(self):
        p, v = self._page("Research components", "Reward modules are trusted executable Python. Metadata discovery is passive; selected modules load in an isolated, time-limited worker.")
        row = QtWidgets.QHBoxLayout(); row.addWidget(self._button("Import selected Python…", self.import_component)); row.addWidget(self._button("Refresh", self.refresh_components)); row.addStretch(); v.addLayout(row)
        self.component_table = QtWidgets.QTableWidget(0, 6); self.component_table.setHorizontalHeaderLabels(["Name", "Identifier", "Compatibility", "Prior experiment use", "Source", "Source hash"]); self.component_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch); v.addWidget(self.component_table, 1)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self._button("Inspect / synthetic validate", self.validate_component)); row.addWidget(self._button("Use in selected experiment", self.apply_component)); v.addLayout(row)
        self.component_report = self._log(v)

    def _make_training(self):
        p, v = self._page("Training", "Review the exact launch preview before starting. Runtime operations are supervised; stop sends a graceful interrupt to the selected process.")
        form = QtWidgets.QFormLayout(); v.addLayout(form)
        self.train_config = QtWidgets.QLineEdit(); self.train_run = QtWidgets.QLineEdit(); self.train_updates = QtWidgets.QSpinBox(); self.train_updates.setRange(1, 10_000_000); self.train_updates.setValue(3)
        self.train_config.editingFinished.connect(lambda:self._set_terrain_default(self.train_config.text()) if self.train_config.text().strip() else None)
        self.train_envs = QtWidgets.QSpinBox(); self.train_envs.setRange(1, 1024); self.train_envs.setValue(1)
        self.train_terrain = QtWidgets.QComboBox(); self.train_terrain.addItems(["independent", "shared"])
        self.train_kind = QtWidgets.QComboBox(); self.train_kind.addItems(["Research training", "Engineering smoke (explicit)"])
        self.monitor_during_training=QtWidgets.QCheckBox("Refresh metrics while training");self.monitor_during_training.setChecked(True)
        form.addRow("Experiment config", self.train_config); form.addRow("New run directory", self.train_run); form.addRow("PPO optimizer updates", self.train_updates); form.addRow("Parallel environments", self.train_envs); form.addRow("Terrain mode",self.train_terrain); form.addRow("Run label", self.train_kind);form.addRow("Monitoring",self.monitor_during_training)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self._button("Select config…", self.browse_train_config)); row.addWidget(self._button("Preview launch", self.preview_training)); row.addWidget(self._button("Launch training", self.launch_training)); row.addWidget(self._button("Stop active process", self.stop_training)); v.addLayout(row)
        self.training_process = None; self.training_log = self._log(v)

    def _make_runs(self):
        p, v = self._page("Runs and metrics", "Run state is reconstructed from manifests and artifacts. Physics ticks, environment decisions, optimizer updates and wall time have distinct units.")
        row = QtWidgets.QHBoxLayout(); row.addWidget(self._button("Refresh runs", self.refresh_runs)); row.addWidget(self._button("Open run folder", self.open_run_folder)); row.addWidget(self._button("Resume selected…", self.resume_run)); row.addWidget(self._button("Plot metrics", self.plot_run)); v.addLayout(row)
        self.run_table = QtWidgets.QTableWidget(0, 7); self.run_table.setHorizontalHeaderLabels(["Run", "Status", "Update", "Environment decisions", "Return", "Checkpoint", "Path"]); self.run_table.horizontalHeader().setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch); v.addWidget(self.run_table, 1)
        self.run_details = self._log(v); self.run_table.itemSelectionChanged.connect(self.show_run_status)

    def _make_compare(self):
        p, v = self._page("Compare runs", "Comparison preserves each run's own configuration and step-aligned metrics; mismatched reward contracts are called out.")
        row = QtWidgets.QHBoxLayout(); self.compare_ids = QtWidgets.QLineEdit(); self.compare_ids.setPlaceholderText("Two or more run IDs or paths, separated by commas"); row.addWidget(self.compare_ids, 1); row.addWidget(self._button("Compare", self.compare_runs)); v.addLayout(row)
        self.compare_report = self._log(v)

    def _make_replay(self):
        p, v = self._page("Replay", "Replay uses the deterministic simulator action tape; it is not rendered Casualties: Unknown gameplay.")
        row = QtWidgets.QHBoxLayout(); self.replay_id = QtWidgets.QLineEdit(); self.replay_id.setPlaceholderText("Run ID or run path"); row.addWidget(self.replay_id, 1); row.addWidget(self._button("Preview replay", lambda: self.replay_run(True))); row.addWidget(self._button("Run replay", lambda: self.replay_run(False))); v.addLayout(row)
        self.replay_report = self._log(v)

    def _make_environment(self):
        p, v = self._page("Environment and settings", "System/runtime inspection is read-only. The UI interpreter is distinct from the configured Isaac interpreter.")
        row = QtWidgets.QHBoxLayout(); row.addWidget(self._button("Refresh diagnostics", self.refresh_environment)); row.addWidget(self._button("Inspect configured runtime", self.inspect_runtime)); row.addWidget(self._button("Choose Isaac python.sh…", self.select_isaac)); row.addWidget(self._button("Choose run output root…",self.select_run_root)); v.addLayout(row)
        self.environment_report = self._log(v)

    def _make_help(self):
        p, v = self._page("Help", "Five-minute start")
        text = QtWidgets.QLabel("""1. Select Experiments → New experiment and write a question and hypothesis. The scaffold contains engineering values; review every choice.\n2. Open the configuration in the editor. It exposes action heads, observation fields, reward terms, model layers and PPO settings. Python callbacks stay code-defined.\n3. Add a reward component under the Components page. Inspect its JSON metadata and source, configure declared parameters, and run synthetic validation.\n4. Select Training, choose an Isaac python.sh interpreter in Environment settings, set a new run directory, and preview the command. Full simulator training requires Isaac Lab.\n5. Inspect metrics, compare runs, and replay recorded actions from the corresponding pages. Plot and replay path figures open in the studio; replay CSV preserves exact measurements.\n\nCommands without Isaac: python -m casu_grid.workbench.lab guide\nGUI: python -m casu_grid.workbench.gui\nDocs: casu_grid/workbench/README.md and ISAAC_LAB_SETUP.md\n\nThis environment is a source-informed surrogate, not bit-exact Casualties: Unknown. The GUI does not approve research configurations or tune scientific parameters for you.""")
        text.setWordWrap(True); v.addWidget(text); v.addStretch()

    def _error(self, message): QtWidgets.QMessageBox.critical(self, "Research Studio", message)
    def _info(self, message): QtWidgets.QMessageBox.information(self, "Research Studio", message)

    def _start_command(self, args, output, callback=None, environment=None, executable=None):
        worker = CommandWorker(args, self, executable=executable, environment=environment); self._worker = worker
        def finish(code, stdout, stderr):
            output.setPlainText((stdout + ("\n" + stderr if stderr else "")).strip())
            if code and callback is None: self._error(f"Command failed ({code}):\n{stderr or stdout}")
            if callback: callback(code, stdout, stderr)
        worker.done.connect(finish); worker.start()

    def refresh_all(self):
        self.refresh_configs(); self.refresh_components(); self.refresh_runs(); self.refresh_environment(); self._refresh_home()
    def refresh_configs(self):
        rows=[]
        if CONFIGS.is_dir():
            hidden_registry=CONFIGS/".studio_hidden.json"
            try:hidden=set(json.loads(hidden_registry.read_text())) if hidden_registry.exists() else set()
            except Exception:hidden=set()
            for path in sorted(CONFIGS.glob("*.py")):
                if path.name.startswith("_"): continue
                if (not getattr(self,"show_hidden",None) or not self.show_hidden.isChecked()) and str(path.resolve()) in hidden:continue
                try:
                    raw=path.read_bytes();tree=__import__("ast").parse(raw.decode("utf-8"),filename=str(path));meta={}
                    for node in tree.body:
                        if isinstance(node,__import__("ast").Assign) and any(isinstance(t,__import__("ast").Name) and t.id=="EXPERIMENT" for t in node.targets):
                            try:meta=__import__("ast").literal_eval(node.value)
                            except (ValueError,SyntaxError):meta={"name":path.stem,"description":"Python-defined metadata (inspect explicitly)"}
                    rows.append((path,meta or {"name":path.stem},hashlib.sha256(raw).hexdigest(),datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")))
                except Exception as exc: rows.append((path,{"name":path.stem,"description":f"Cannot parse metadata: {exc}"},"invalid","unknown"))
        query=self.config_search.text().casefold() if hasattr(self,"config_search") else ""
        rows=[r for r in rows if query in (str(r[0])+str(r[1])).casefold()]
        self.config_table.setRowCount(len(rows))
        for i,(path,meta,digest,modified) in enumerate(rows):
            associated=[r["run_id"] for r in getattr(self,"_runs",[]) if (meta.get("id") and r.get("experiment_id")==meta.get("id")) or (r.get("config_path") and Path(r["config_path"]).expanduser().resolve()==path.resolve())]
            values=[meta.get("name",path.stem),meta.get("hypothesis",meta.get("description","")),modified,digest,", ".join(associated) or "—",str(path)]
            for j,value in enumerate(values): self.config_table.setItem(i,j,QtWidgets.QTableWidgetItem(str(value)))
        self._configs=rows
    def _selected_config(self):
        row=self.config_table.currentRow()
        return self._configs[row][0] if 0<=row<len(getattr(self,"_configs",[])) else None
    def create_experiment(self):
        name,ok=QtWidgets.QInputDialog.getText(self,"New experiment","Experiment name")
        if not ok or not name.strip(): return
        hypothesis,ok=QtWidgets.QInputDialog.getText(self,"Research hypothesis","Question or hypothesis (optional)")
        if not ok:return
        dest, _=QtWidgets.QFileDialog.getSaveFileName(self,"Create Python experiment",str(CONFIGS/(name.strip().lower().replace(" ","_")+".py")),"Python (*.py)")
        if not dest:return
        try: lab.scaffold(name.strip(),Path(dest),hypothesis=hypothesis)
        except Exception as exc:self._error(str(exc));return
        self.refresh_configs(); self.editor_path.setText(str(Path(dest))); self.load_editor(); self.nav.setCurrentRow(2)
    def open_selected_config(self):
        path=self._selected_config()
        if path:self.editor_path.setText(str(path));self.load_editor();self.nav.setCurrentRow(2)
    def duplicate_config(self):
        path=self._selected_config()
        if not path:return self._error("Select an experiment first.")
        dest,_=QtWidgets.QFileDialog.getSaveFileName(self,"Duplicate configuration",str(path.with_name(path.stem+"_copy.py")),"Python (*.py)")
        if not dest:return
        try:
            if Path(dest).exists():raise ValueError(f"Refusing to overwrite {dest}")
            source=path.read_text(); destpath=Path(dest)
            original_meta=next((meta for p,meta,*_ in self._configs if p==path),{})
            identity={"id":uuid.uuid4().hex,"name":Path(dest).stem.replace("_"," "),"description":f"Copy of {original_meta.get('name',path.stem)}","hypothesis":original_meta.get("hypothesis",""),"notes":original_meta.get("notes",""),"parent":original_meta.get("id"),"created_utc":datetime.now(timezone.utc).isoformat(),"config_version":1}
            destpath.parent.mkdir(parents=True,exist_ok=True)
            destpath.write_text(source+"\n# New identity for this duplicated research configuration.\nEXPERIMENT = "+repr(identity)+"\n",encoding="utf-8")
            self.refresh_configs(); self.editor_path.setText(str(destpath)); self.load_editor()
        except Exception as exc:self._error(str(exc))
    def toggle_hidden_config(self):
        path=self._selected_config()
        if not path:return self._error("Select an experiment first.")
        registry=CONFIGS/".studio_hidden.json"
        try:hidden=set(json.loads(registry.read_text())) if registry.exists() else set()
        except Exception as exc:return self._error(f"Cannot read hide registry: {exc}")
        key=str(path.resolve())
        if key in hidden:hidden.remove(key);message="Experiment restored to the normal library."
        else:hidden.add(key);message="Experiment hidden from the normal library. Its Python source and run history remain untouched."
        registry.write_text(json.dumps(sorted(hidden),indent=2)+"\n",encoding="utf-8")
        self.refresh_configs();self.statusBar().showMessage(message,5000)
    def browse_config(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Open experiment configuration",str(CONFIGS),"Python (*.py)")
        if path:self.editor_path.setText(path);self.load_editor()
    def load_editor(self):
        try:
            path=Path(self.editor_path.text()).expanduser().resolve(); raw=path.read_bytes(); self.editor.setPlainText(raw.decode("utf-8")); self._editor_original=raw;self._editor_digest=hashlib.sha256(raw).hexdigest()
            self.editor_hash.setText(f"Loaded {path} · SHA-256 {self._editor_digest}"); self.editor_report.clear()
        except Exception as exc:self._error(str(exc))
    def review_editor(self):
        import difflib
        if self._editor_original is None:return self._error("Load a configuration first.")
        before=self._editor_original.decode("utf-8").splitlines(True);after=self.editor.toPlainText().splitlines(True)
        self.editor_report.setPlainText("".join(difflib.unified_diff(before,after,fromfile="saved",tofile="unsaved")) or "No changes.")
    def save_editor(self):
        if self._editor_original is None:return self._error("Load a configuration first.")
        path=Path(self.editor_path.text()).expanduser().resolve()
        try:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self._editor_digest:raise ValueError("File changed on disk since it was loaded. Reload and review before saving.")
            updated=self.editor.toPlainText().encode("utf-8")
            import ast; ast.parse(updated.decode("utf-8"),filename=str(path))
            if self._editor_original!=updated:
                import difflib
                diff="".join(difflib.unified_diff(self._editor_original.decode("utf-8").splitlines(True),updated.decode("utf-8").splitlines(True),fromfile="saved",tofile="proposed"))
                dialog=QtWidgets.QMessageBox(self);dialog.setWindowTitle("Review configuration changes");dialog.setText("Save these explicit Python source changes?");dialog.setDetailedText(diff or "No changes.");dialog.setStandardButtons(QtWidgets.QMessageBox.Save|QtWidgets.QMessageBox.Cancel)
                if dialog.exec()!=QtWidgets.QMessageBox.Save:return
            path.write_bytes(updated); self._editor_original=updated;self._editor_digest=hashlib.sha256(updated).hexdigest();self.editor_hash.setText(f"Saved · SHA-256 {self._editor_digest}");self.refresh_configs()
        except Exception as exc:self._error(str(exc))
    def inspect_editor(self):
        try:self.editor_report.setPlainText(json.dumps(lab.inspect_config(Path(self.editor_path.text())),indent=2,default=str))
        except Exception as exc:self.editor_report.setPlainText(f"Contract inspection failed: {exc}")
    def configure_model_ppo(self):
        if self._editor_original is None:return self._error("Load an experiment configuration first.")
        if self.editor.toPlainText().encode()!=self._editor_original:return self._error("Save or discard current source edits before using structured controls.")
        try:
            path=Path(self.editor_path.text()).expanduser().resolve();module,_=lab._load_config(path)
            dialog=ContractDialog(module.MODEL,module.PPO,self)
            if dialog.exec()!=QtWidgets.QDialog.Accepted:return
            sections=dialog.values()
            from pprint import pformat
            text=self.editor.toPlainText()+"\n# Structured Research Studio contract update (review before saving).\nfrom casu_grid.workbench.contracts import ModelContract, PPOContract\nMODEL = ModelContract(**"+pformat(sections["MODEL"],sort_dicts=True)+")\nPPO = PPOContract(**"+pformat(sections["PPO"],sort_dicts=True)+")\n"
            self.editor.setPlainText(text);self.review_editor();self.statusBar().showMessage("Review the generated contract diff, then save explicitly.",8000)
        except Exception as exc:self._error(f"Could not edit typed contracts: {exc}")
    def probe_observations(self):
        path=Path(self.editor_path.text()).expanduser() if self.editor_path.text().strip() else None
        if not path or not path.is_file():return self._error("Load a valid experiment config first.")
        try:isaac,env=self._isaac_environment()
        except Exception as exc:return self._error(str(exc))
        self._start_command(["observation-probe",str(path.resolve())],self.editor_report,environment=env,executable=isaac)
    def browse_train_config(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Training configuration",str(CONFIGS),"Python (*.py)")
        if path:
            self.train_config.setText(path); self.train_run.setText(str(self.run_root/Path(path).stem))
            self._set_terrain_default(path)
    def _set_terrain_default(self,path):
        try:
            config,_=lab._load_config(Path(path))
            self.train_terrain.setCurrentText(lab.configured_terrain_mode(config))
        except Exception:
            self.train_terrain.setCurrentText("shared")
    def _launch_args(self,preview):
        if not self.train_config.text().strip() or not self.train_run.text().strip():raise ValueError("Select an experiment config and a new run directory.")
        args=["--experiments",str(self.run_root),"launch",self.train_config.text(),"--run-dir",self.train_run.text(),"--updates",str(self.train_updates.value()),"--num-envs",str(self.train_envs.value()),"--terrain-mode",self.train_terrain.currentText()]
        if preview:args.append("--preview")
        if self.train_kind.currentIndex()==1:args.append("--engineering-smoke")
        return args
    def _isaac_environment(self):
        env=os.environ.copy();sim=Path(env.get("ISAAC_SIM", ""))
        if not sim.is_file():raise ValueError("Select the Isaac Sim python.sh file in Environment settings.")
        root=sim.parent.parent;lab_root=root/"IsaacLab-2.1.1"
        env["LD_LIBRARY_PATH"]=str(root/"compat-libs/usr/lib64")+os.pathsep+env.get("LD_LIBRARY_PATH","")
        support=[root/"isaaclab-deps",lab_root/"source/isaaclab",lab_root/"source/isaaclab_tasks",lab_root/"source/isaaclab_assets"]
        env["PYTHONPATH"]=os.pathsep.join(str(x) for x in support if x.exists())+os.pathsep+env.get("PYTHONPATH","")
        env["OMNI_KIT_ACCEPT_EULA"]="YES";env["HEADLESS"]="1"
        env["HOME"]=str(root/"user-home");env["XDG_CACHE_HOME"]=str(root/"cache");env["XDG_CONFIG_HOME"]=str(root/"config")
        return str(sim),env
    def preview_training(self):
        try:self._start_command(self._launch_args(True),self.training_log)
        except Exception as exc:self._error(str(exc))
    def launch_training(self):
        try:
            config=Path(self.train_config.text()).resolve(); summary=lab.inspect_config(config)
            if not summary["callbacks_defined"]:raise ValueError("Training config requires selected callbacks and explicit reward code.")
            self._isaac_environment()
            preview_args=self._launch_args(True)
            isaac,env=self._isaac_environment()
            self._start_command(preview_args,self.training_log,lambda code,out,err:self._confirm_launch(code,out,err),environment=env,executable=isaac)
        except Exception as exc:self._error(str(exc))
    def _confirm_launch(self,code,out,err):
        if code:return self._error(err or out)
        if QtWidgets.QMessageBox.question(self,"Confirm training launch",out+"\n\nStart this process now?")!=QtWidgets.QMessageBox.Yes:return
        args=self._launch_args(False); self.training_log.appendPlainText("\nStarting supervised launch…")
        # CLI launcher supervises trainer; Popen is retained so Stop can interrupt it.
        try:
            isaac,env=self._isaac_environment()
            self.training_process=subprocess.Popen([isaac,"-m","casu_grid.workbench.lab",*args],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            self.training_log.appendPlainText(f"Launcher PID {self.training_process.pid}; see run artifacts for metrics and checkpoint state.")
            self._training_timer=QtCore.QTimer(self);self._training_timer.timeout.connect(self._refresh_training_status);self._training_timer.start(1200)
        except Exception as exc:self._error(str(exc))
    def _poll_training(self):
        proc=self.training_process
        if proc and proc.poll() is not None:
            failed=lab.training_result_failed(proc.returncode,Path(self.train_run.text()).expanduser())
            if failed:
                message=f"TRAINING FAILED (launcher exit status {proc.returncode}; see failure.json and launcher.log)."
                self.training_log.appendPlainText(message);self.statusBar().showMessage(message,15000)
                self._error(message)
            elif Path(self.train_run.text()).expanduser().joinpath("INTERRUPTED").is_file():
                self.training_log.appendPlainText("Training interrupted; inspect the run checkpoint before resuming.")
                self.statusBar().showMessage("Training interrupted.",10000)
            else:
                self.training_log.appendPlainText(f"Training completed (launcher exit status {proc.returncode}).")
                self.statusBar().showMessage("Training completed successfully.",10000)
            self.training_process=None;self.refresh_runs()
    def _refresh_training_status(self):
        proc=self.training_process
        if not proc:return
        try:
            if proc.stdout:
                import select
                while select.select([proc.stdout],[],[],0)[0]:
                    line=proc.stdout.readline()
                    if not line:break
                    self.training_log.appendPlainText(line.rstrip())
        except Exception:pass
        if self.monitor_during_training.isChecked():self.refresh_runs()
        self._poll_training()
        if not self.training_process and hasattr(self,"_training_timer"):self._training_timer.stop()
    def stop_training(self):
        if self.training_process and self.training_process.poll() is None:
            import signal
            os.killpg(self.training_process.pid,signal.SIGINT);self.training_log.appendPlainText("Interrupt sent to the supervised simulator process group; waiting for checkpoint handling.")
        else:self._info("No active training process owned by this GUI session.")
    def refresh_runs(self):
        if not hasattr(self,"run_table"):return
        try:rows=lab.discover_runs(self.run_root)
        except Exception as exc:self.run_details.setPlainText(str(exc));return
        self._runs=rows; self.run_table.setRowCount(len(rows))
        for i,r in enumerate(rows):
            values=[r.get("run_id"),r.get("status"),r.get("update"),r.get("environment_steps"),r.get("mean_completed_return"),"yes" if r.get("latest_checkpoint") else "no",r.get("path")]
            for j,value in enumerate(values):self.run_table.setItem(i,j,QtWidgets.QTableWidgetItem("—" if value is None else str(value)))
        if hasattr(self,"config_table"):self.refresh_configs()
        if hasattr(self,"home_recent"):self._refresh_home()
    def _selected_run(self):
        row=self.run_table.currentRow() if hasattr(self,"run_table") else -1
        return self._runs[row] if 0<=row<len(getattr(self,"_runs",[])) else None
    def show_run_status(self):
        run=self._selected_run()
        if run:self.run_details.setPlainText(json.dumps(run,indent=2))
    def open_run_folder(self):
        run=self._selected_run()
        if run:QtCore.QUrl.fromLocalFile(run["path"]);QtGui=__import__("PySide6.QtGui",fromlist=["QDesktopServices"]);QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(run["path"]))
    def resume_run(self):
        run=self._selected_run()
        if not run:return self._error("Select a run first.")
        updates,ok=QtWidgets.QInputDialog.getInt(self,"Resume checkpoint","Additional PPO optimizer updates",3,1,10_000_000)
        if not ok:return
        output=QtWidgets.QFileDialog.getExistingDirectory(self,"Choose parent for a new resume run",str(self.run_root))
        if not output:return
        target=Path(output)/(Path(run["path"]).name+"-resume")
        isaac,env=self._isaac_environment()
        self._start_command(["--experiments",str(self.run_root),"resume",run["run_id"],"--updates",str(updates),"--output",str(target),"--preview"],self.run_details,lambda c,o,e:self._confirm_resume(c,o,e,run,updates,target),environment=env,executable=isaac)
    def _confirm_resume(self,code,out,err,run,updates,target):
        self.run_details.setPlainText(out+"\n"+err)
        if code:return
        if QtWidgets.QMessageBox.question(self,"Resume PPO",out+"\n\nRun this resume command?")==QtWidgets.QMessageBox.Yes:
            isaac,env=self._isaac_environment()
            self._start_command(["--experiments",str(self.run_root),"resume",run["run_id"],"--updates",str(updates),"--output",str(target)],self.run_details,environment=env,executable=isaac)
    def plot_run(self):
        run=self._selected_run()
        if run:
            image=Path(run["path"])/"analysis"/"training_metrics.png"
            if image.exists() and QtWidgets.QMessageBox.question(self,"Replace generated plot?",f"This will replace the derived plot file:\n{image}")!=QtWidgets.QMessageBox.Yes:return
            def plotted(code,stdout,stderr):
                if code:return self._error(stderr or stdout)
                if image.is_file():self.show_plot(image,"Training metrics · "+run["run_id"])
                else:self._error(f"Plot command completed but image is missing: {image}\n{stdout}\n{stderr}")
            if __import__("importlib.util",fromlist=["find_spec"]).find_spec("matplotlib"):
                self._start_command(["--experiments",str(self.run_root),"plot",run["run_id"]],self.run_details,plotted)
            elif os.environ.get("ISAAC_SIM"):
                isaac,env=self._isaac_environment();self._start_command(["--experiments",str(self.run_root),"plot",run["run_id"]],self.run_details,plotted,environment=env,executable=isaac)
            else:self._error("Matplotlib is not installed in the GUI Python. Install it or configure the Isaac runtime in Environment settings.")
    def show_plot(self,path,title):
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle(title);dialog.resize(900,650);layout=QtWidgets.QVBoxLayout(dialog)
        image=QtGui.QPixmap(str(path))
        if image.isNull():return self._error(f"Could not open plot image: {path}")
        label=QtWidgets.QLabel();label.setAlignment(QtCore.Qt.AlignCenter);label.setPixmap(image.scaled(860,560,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation));layout.addWidget(label)
        layout.addWidget(QtWidgets.QLabel(f"Source: {path}"));button=QtWidgets.QPushButton("Close");button.clicked.connect(dialog.accept);layout.addWidget(button);dialog.exec()
    def compare_runs(self):
        values=[x.strip() for x in self.compare_ids.text().split(",") if x.strip()]
        if len(values)<2:return self._error("Enter at least two run IDs or paths.")
        self._start_command(["--experiments",str(self.run_root),"compare",*values,"--json"],self.compare_report)
    def replay_run(self,preview):
        if not self.replay_id.text().strip():return self._error("Enter a run ID or path.")
        if not preview:
            try:
                selected=lab._find_run(self.replay_id.text().strip(),self.run_root);existing=[selected/"replay"/name for name in ("replay.csv","paths.png","summary.json") if (selected/"replay"/name).exists()]
                if existing and QtWidgets.QMessageBox.question(self,"Replace replay artifacts?","This replay will replace derived files:\n"+"\n".join(str(path) for path in existing))!=QtWidgets.QMessageBox.Yes:return
            except Exception as exc:return self._error(str(exc))
        args=["--experiments",str(self.run_root),"replay",self.replay_id.text().strip()]
        if preview:args.append("--preview")
        if not preview:
            isaac,env=self._isaac_environment()
            def replayed(code,stdout,stderr):
                if code:return self._error(stderr or stdout)
                try:run=lab._find_run(self.replay_id.text().strip(),self.run_root)
                except Exception as exc:return self._error(str(exc))
                image=run/"replay"/"paths.png"
                if image.is_file():self.show_plot(image,"Recorded simulator trajectory · "+run.name)
                else:self._error(f"Replay plot is missing: {image}\n{stdout}\n{stderr}")
            self._start_command(args,self.replay_report,replayed,environment=env,executable=isaac)
        else:self._start_command(args,self.replay_report)
    def refresh_components(self):
        if not hasattr(self,"component_table"):return
        try:COMPONENTS.mkdir(parents=True,exist_ok=True);rows=components.discover_components(COMPONENTS)
        except Exception as exc:self.component_report.setPlainText(str(exc));return
        self._components=rows;self.component_table.setRowCount(len(rows))
        for i,r in enumerate(rows):
            m=r.get("metadata",{});uses=self._component_usage(m.get("id"));use_label=", ".join(uses) if uses else "Not referenced"
            values=[m.get("name",r.get("error","Invalid metadata")),m.get("id","—"),m.get("environment_compatibility","declared by author"),use_label,r.get("source","") ,r.get("sha256","")]
            for j,value in enumerate(values):
                cell=QtWidgets.QTableWidgetItem(str(value))
                if j==3 and uses:cell.setToolTip("\n".join(uses))
                self.component_table.setItem(i,j,cell)
    def _component_usage(self,component_id):
        if not component_id or not CONFIGS.is_dir():return []
        ast=__import__("ast");uses=[]
        for path in CONFIGS.glob("*.py"):
            try:
                tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
                for node in tree.body:
                    if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=="REWARD_COMPONENT_REF" for target in node.targets):
                        ref=ast.literal_eval(node.value)
                        if isinstance(ref,dict) and ref.get("id")==component_id:uses.append(str(ref.get("name") or path.stem))
                        break
            except (OSError,SyntaxError,ValueError,TypeError):continue
        return uses
    def _selected_component(self):
        row=self.component_table.currentRow();return self._components[row] if 0<=row<len(getattr(self,"_components",[])) else None
    def import_component(self):
        source,_=QtWidgets.QFileDialog.getOpenFileName(self,"Select trusted Python reward component",str(ROOT),"Python (*.py)")
        if not source:return
        if QtWidgets.QMessageBox.warning(self,"Trust selected Python code",f"Validation will import and execute this source in a subprocess:\n{source}\n\nContinue only if you trust it.",QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.Cancel)!=QtWidgets.QMessageBox.Yes:return
        try:
            import importlib.util
            spec=importlib.util.spec_from_file_location("_reward_metadata_discovery",source)
            # Avoid importing for discovery: require an adjacent JSON declaration.
            meta_path=Path(str(source)+".component.json")
            if not meta_path.is_file():raise ValueError("Create adjacent .py.component.json metadata first. See the component author guide in the workbench README.")
            meta=json.loads(meta_path.read_text()); dest=COMPONENTS/Path(source).name; dest_meta=Path(str(dest)+".component.json")
            if dest.exists() or dest_meta.exists():raise ValueError("A component with this filename is already registered; choose a distinct source filename.")
            import shutil;shutil.copy2(source,dest);copied=dict(meta);copied["source_sha256"]=hashlib.sha256(dest.read_bytes()).hexdigest();dest_meta.write_text(json.dumps(copied,indent=2)+"\n")
            self.refresh_components();self._info(f"Imported explicitly selected component to {dest}\nReview metadata and run synthetic validation before use.")
        except Exception as exc:self._error(str(exc))
    def validate_component(self):
        item=self._selected_component()
        if not item:return self._error("Select a component first.")
        try:
            meta=item["metadata"]
            params={}
            for field in meta.get("parameter_schema",[]):
                default=field.get("default")
                kind=field.get("value_type") if field.get("type")=="optional" else field.get("type")
                if field.get("type")=="optional":
                    raw,ok=QtWidgets.QInputDialog.getText(self,"Optional component parameter",field["name"]+" (blank means None)",text="" if default is None else str(default))
                    if not ok:return
                    if not raw.strip():value=None
                    elif kind=="boolean":value=raw.strip().lower() in {"true","1","yes"}
                    elif kind=="integer":value=int(raw)
                    elif kind=="float":value=float(raw)
                    elif kind=="enum":value=raw
                    else:raise ValueError(f"Unsupported optional value type: {kind}")
                elif kind=="boolean":value,ok=QtWidgets.QInputDialog.getItem(self,"Component parameter",field["name"],["true","false"],0 if default else 1,False);value=value=="true"
                elif kind=="enum":value,ok=QtWidgets.QInputDialog.getItem(self,"Component parameter",field["name"],field.get("choices",[]),0,False)
                elif kind=="integer":value,ok=QtWidgets.QInputDialog.getInt(self,"Component parameter",field["name"],int(default or 0),int(field.get("minimum",-2147483647)),int(field.get("maximum",2147483647)))
                elif kind=="float":value,ok=QtWidgets.QInputDialog.getDouble(self,"Component parameter",field["name"],float(default or 0),float(field.get("minimum",-1e100)),float(field.get("maximum",1e100)),6)
                else:value=json.loads(QtWidgets.QInputDialog.getText(self,"Component parameter",field["name"],text=json.dumps(default))[0] or "null");ok=True
                if not ok:return
                params[field["name"]]=value
            result=components.validate_component(Path(item["source"]),meta,params)
            self._validated_component=(item,result)
            self.component_report.setPlainText("Synthetic worker validation passed. This does not establish compatibility with live Isaac state.\n"+json.dumps(result,indent=2))
        except Exception as exc:self.component_report.setPlainText(f"Validation failed: {exc}")
    def apply_component(self):
        data=getattr(self,"_validated_component",None)
        if not data:return self._error("Run synthetic validation first.")
        item,result=data;path,ok=QtWidgets.QFileDialog.getOpenFileName(self,"Select experiment to configure",str(CONFIGS),"Python (*.py)")
        if not path or not ok:return
        try:
            if hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest()!=result["source_sha256"]:
                raise ValueError("Component source changed after validation. Refresh the registry and validate again.")
            target_module,_=lab._load_config(Path(path))
            current_names=sorted(term.name for term in target_module.REWARD.terms)
            component_names=sorted(result["metadata"].get("reward_terms",result["synthetic_result"][1].keys()))
            if current_names!=component_names:
                raise ValueError(f"Experiment RewardContract terms {current_names} do not match this component's output terms {component_names}. Update the Python REWARD contract explicitly, validate it, then select the component again.")
            p=Path(path);raw=p.read_bytes();source=raw.decode("utf-8");marker="REWARD_COMPONENT_REF = "
            if marker in source:raise ValueError("This config already declares REWARD_COMPONENT_REF; remove/review the existing selection explicitly before replacing it.")
            ref={"id":result["metadata"]["id"],"source":os.path.relpath(item["source"],p.parent),"entry_point":result["metadata"].get("entry_point","reward"),"source_sha256":result["source_sha256"],"contract_version":1,"parameters":result["parameters"],"reward_terms":result["metadata"].get("reward_terms",sorted(result["synthetic_result"][1])),"validation":"synthetic-only","validated_sha256":result["source_sha256"]}
            updated=(source+"\n# Researcher-selected reward component provenance.\nREWARD_COMPONENT_REF = "+repr(ref)+"\n").encode()
            if hashlib.sha256(p.read_bytes()).digest()!=hashlib.sha256(raw).digest():raise ValueError("Experiment changed externally during component selection; reload it before applying.")
            import ast;ast.parse(updated.decode(),filename=str(p))
            diff="".join(__import__("difflib").unified_diff(source.splitlines(True),updated.decode().splitlines(True),fromfile="saved",tofile="component selection"))
            dialog=QtWidgets.QMessageBox(self);dialog.setWindowTitle("Apply reward component");dialog.setText("Add this component reference to the experiment configuration?");dialog.setDetailedText(diff);dialog.setStandardButtons(QtWidgets.QMessageBox.Save|QtWidgets.QMessageBox.Cancel)
            if dialog.exec()!=QtWidgets.QMessageBox.Save:return
            p.write_bytes(updated)
            self._info("Component source, entry point, parameters, reward term names and source hash are now recorded in the selected experiment. Training and replay resolve the callback from that reference without replacing action or observation callbacks. Live state compatibility still requires an Isaac probe.")
            self.refresh_configs()
        except Exception as exc:self._error(str(exc))
    def refresh_environment(self):
        if not hasattr(self,"environment_report"):return
        import platform, shutil
        isaac=os.environ.get("ISAAC_SIM","");p=Path(isaac) if isaac else None
        git=subprocess.run(["git","-C",str(ROOT),"rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
        status=subprocess.run(["git","-C",str(ROOT),"status","--short"],capture_output=True,text=True).stdout.strip()
        info={"project_root":str(ROOT),"revision":git,"worktree_changes":status.splitlines(),"ui_python":sys.executable,"python_version":sys.version,"platform":platform.platform(),"isaac_python_sh":isaac or None,"isaac_runtime_available":bool(p and p.is_file()),"component_library":str(COMPONENTS),"experiment_config_root":str(CONFIGS),"runs_root":str(self.run_root),"physics_timestep_seconds":0.02,"physics_ticks_per_second":50,"physics_ticks_per_policy_decision":10,"canonical_seed":391728405}
        self.environment_report.setPlainText(json.dumps(info,indent=2))
    def inspect_runtime(self):
        try:executable,env=self._isaac_environment()
        except Exception as exc:return self._error(str(exc))
        self.environment_report.appendPlainText("\nQuerying configured bundled Python package/GPU information…")
        worker=RuntimeInfoWorker(executable,env,self);worker.done.connect(lambda result:self.environment_report.appendPlainText("\n"+result));worker.start();self._runtime_worker=worker
    def select_isaac(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Select Isaac Sim python.sh",str(Path.home()),"python.sh (python.sh)")
        if path:
            os.environ["ISAAC_SIM"]=path
            self.settings.setValue("isaac_sim",path)
            QtWidgets.QMessageBox.information(self,"Isaac interpreter selected","Stored for this GUI session. For CLI use, export ISAAC_SIM to this path in your shell.")
            self.refresh_environment()
    def select_run_root(self):
        path=QtWidgets.QFileDialog.getExistingDirectory(self,"Choose experiment run output root",str(self.run_root))
        if path:
            self.run_root=Path(path).expanduser().resolve();self.settings.setValue("run_root",str(self.run_root))
            self.statusBar().showMessage(f"Run output root: {self.run_root}",7000);self.refresh_runs();self.refresh_environment()
    def _refresh_home(self):
        if not hasattr(self,"home_info"):return
        self.home_info.setText(f"Project: {ROOT}\nEnvironment: source-validated Isaac Layer 1 surrogate · canonical seed 391728405\nPhysics: 0.02 s / 50 ticks per second · 10 physics ticks per policy decision\nIsaac python.sh: {os.environ.get('ISAAC_SIM','not configured')}\nRun output root: {self.run_root}\nRuns: {len(getattr(self,'_runs',[]))} · active: {sum(r['status']=='running' for r in getattr(self,'_runs',[]))}")
        lines=[]
        for path,meta,digest,*_ in getattr(self,"_configs",[])[:6]:lines.append(f"Experiment · {meta.get('name',path.stem)} · {path}")
        for run in getattr(self,"_runs",[])[:8]:lines.append(f"Run · {run['run_id']} · {run['status']} · update {run.get('update')}")
        self.home_recent.setPlainText("\n".join(lines) or "No experiment configurations or runs found. Create an explicitly labeled engineering experiment to explore the workbench.")
    def closeEvent(self,event):
        self.settings.setValue("geometry",self.saveGeometry())
        active_worker=getattr(self,"_worker",None)
        runtime_worker=getattr(self,"_runtime_worker",None)
        if (active_worker and active_worker.isRunning()) or (runtime_worker and runtime_worker.isRunning()):
            self.statusBar().showMessage("A supervised workbench operation is still running. Close after it finishes.",10000)
            event.ignore();return
        if self.training_process and self.training_process.poll() is None:
            answer=QtWidgets.QMessageBox.question(self,"Training is active","Training launched by this window is still active. Close the studio and leave it running?")
            if answer!=QtWidgets.QMessageBox.Yes:event.ignore();return
        super().closeEvent(event)


def main() -> int:
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen" and "--smoke" in sys.argv:
        app=QtWidgets.QApplication(sys.argv[:1]);window=Studio();window.refresh_all();window.close();return 0
    app=QtWidgets.QApplication(sys.argv);window=Studio();window.show();return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
