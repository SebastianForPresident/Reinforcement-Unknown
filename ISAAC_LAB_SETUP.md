# Local Isaac Lab installation

The host's tested Isaac stack is kept outside this repository at:

```text
~/Applications/IsaacLab-2.1.1-IsaacSim-4.5.0/
```

It contains Isaac Sim 4.5.0, Isaac Lab 2.1.1, the Isaac Sim bundled Python
3.10 runtime, and the Fedora `libcrypt.so.1` compatibility library used by
that runtime. The Python support packages needed by the current headless
DirectRLEnv smoke probe are installed in `isaaclab-deps/` beside the two
checkouts. This is a machine-local installation and should not be committed.

Run an Isaac Lab Python script from this repository with:

```bash
ISAAC_ROOT="$HOME/Applications/IsaacLab-2.1.1-IsaacSim-4.5.0"
ISAAC_SIM="$ISAAC_ROOT/isaac-sim-4.5.0"
ISAAC_LAB="$ISAAC_ROOT/IsaacLab-2.1.1"

export LD_LIBRARY_PATH="$ISAAC_ROOT/compat-libs/usr/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ISAAC_ROOT/isaaclab-deps:$ISAAC_LAB/source/isaaclab:$ISAAC_LAB/source/isaaclab_tasks:$ISAAC_LAB/source/isaaclab_assets${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export HEADLESS=1
export HOME="$ISAAC_ROOT/user-home"
export XDG_CACHE_HOME="$ISAAC_ROOT/cache"
export XDG_CONFIG_HOME="$ISAAC_ROOT/config"

"$ISAAC_SIM/python.sh" path/to/script.py --headless
```

The bundled Python is the interpreter to use; system Python 3.14 is not
compatible with this Isaac Sim release. For example, the preserved local
primitive-scene probe can be rerun with:

```bash
"$ISAAC_SIM/python.sh" "$ISAAC_ROOT/smoke.py" \
  --num_envs 16 --warmup 10 --steps 10 --headless
```

On 2026-10-12 that probe launched on `cuda:0` (RTX 3070), created a 16-env
DirectRLEnv, and completed its control steps. The earlier 65,536-env result
is only a trivial-scene throughput ceiling; it does not validate CasU
navigation fidelity. Isaac Sim emitted non-blocking warnings for unavailable
OmniHub/MaterialX and the unset environment seed.
