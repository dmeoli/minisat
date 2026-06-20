from .minisat.gym.MiniSATEnv import gym_sat_Env

try:  # prefer the maintained gymnasium; fall back to legacy gym
    import gymnasium as gym
except ImportError:
    import gym


def register(id, entry_point, force=True):
    # gym < 0.21 exposes `gym.envs.registry.env_specs`; gym >= 0.21 (and gymnasium)
    # make `gym.envs.registry` itself a dict-like {id: EnvSpec}. Support both.
    try:
        registry = gym.envs.registry.env_specs
    except AttributeError:
        registry = gym.envs.registry
    if id in registry:
        if not force:
            return
        registry.pop(id, None)
    gym.register(
        id=id,
        entry_point=entry_point
    )


# Graph-Q-SAT UPD: register the sat environment
register(id="sat-v0", entry_point="minisat.minisat.gym.MiniSATEnv:gym_sat_Env")
