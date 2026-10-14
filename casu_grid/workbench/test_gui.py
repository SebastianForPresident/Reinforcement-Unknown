from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets
from . import gui
from .contracts import ModelContract, PPOContract


class ResearchStudioGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test-gui"])

    def test_window_constructs_and_experiment_discovery_does_not_execute_python(self):
        old_configs,old_components=gui.CONFIGS,gui.COMPONENTS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);configs=root/"experiments";components=root/"components"
            configs.mkdir();sentinel=root/"executed.txt"
            config=configs/"candidate.py"
            config.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\nEXPERIMENT = {{'id':'one','name':'Candidate','hypothesis':'inspect safely'}}\n")
            gui.CONFIGS=configs;gui.COMPONENTS=components
            window=gui.Studio()
            try:
                self.assertEqual(window.nav.count(),10)
                self.assertFalse(sentinel.exists())
                self.assertEqual(len(window._configs),1)
                self.assertEqual(window._configs[0][1]["name"],"Candidate")
                self.assertTrue(components.is_dir())
                window.run_root=root/"results";window.train_config.setText(str(config));window._set_terrain_default(config);window.train_run.setText(str(root/"results"/"trial"));window.train_kind.setCurrentIndex(1)
                command=window._launch_args(True)
                self.assertEqual(command[:3],["--experiments",str(root/"results"),"launch"])
                self.assertIn("--engineering-smoke",command)
                self.assertEqual(command[command.index("--terrain-mode")+1],"shared")
                dialog=gui.ContractDialog(ModelContract(),PPOContract())
                self.assertEqual(dialog.values()["PPO"]["target_kl"],0.03)
                dialog.close()
            finally:
                window.close();gui.CONFIGS=old_configs;gui.COMPONENTS=old_components

    def test_xy1_selection_defaults_training_terrain_to_independent(self):
        window=gui.Studio()
        try:
            config=Path(gui.ROOT)/"casu_grid/workbench/experiments/xy1_point_navigation.py"
            window.train_config.setText(str(config));window.train_run.setText(str(Path(gui.ROOT)/"experiments/xy1-gui-test"));window._set_terrain_default(config)
            self.assertEqual(window.train_terrain.currentText(),"independent")
            window.train_envs.setValue(16);window.train_updates.setValue(3)
            args=window._launch_args(True)
            self.assertEqual(args[args.index("--terrain-mode")+1],"independent")
            self.assertEqual(args[args.index("--num-envs")+1],"16")
            self.assertEqual(args[args.index("--updates")+1],"3")
        finally:window.close()

    def test_typed_contract_dialog_preserves_model_and_ppo_values(self):
        dialog=gui.ContractDialog(ModelContract(),PPOContract())
        values=dialog.values()
        self.assertEqual(values["MODEL"]["spatial_channels"],(16,32))
        self.assertEqual(values["PPO"]["rollout_steps"],256)
        dialog.close()

    def test_plot_viewer_reads_generated_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            from PySide6 import QtGui
            path=Path(tmp)/"plot.png";pixmap=QtGui.QPixmap(32,24);pixmap.fill(QtGui.QColor("white"));self.assertTrue(pixmap.save(str(path)))
            window=gui.Studio();window.show_plot=lambda image,title: self.assertTrue(Path(image).is_file())
            self.assertTrue(Path(path).is_file());window.close()


if __name__ == "__main__":
    unittest.main()
