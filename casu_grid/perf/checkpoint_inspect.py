"""Print the integrity summary of a PPO checkpoint (used by resume_test_opt4.sh)."""
import json, sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
finite = all(bool(torch.isfinite(v).all()) for v in c["model"].values() if v.is_floating_point())
opt = c["optimizer"]["state"]
steps = sorted({int(s["step"]) for s in opt.values()}) if opt else []
env = c.get("environment_state")
def keys(x):
    return sorted(x) if isinstance(x, dict) else type(x).__name__
print(json.dumps({"update_count": c["update_count"], "total_steps": c["total_steps"], "weights_finite": finite,
                  "optimizer_param_states": len(opt), "optimizer_steps": steps[:3],
                  "env_state_keys": keys(env) if not isinstance(env, (list, tuple)) else f"list[{len(env)}]",
                  "observation_present": c.get("observation") is not None}, sort_keys=True))
