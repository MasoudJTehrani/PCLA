"""Cross-agent module isolation for PCLA's in-process agent loader.

PCLA loads every agent into the *same* Python process. Agents are third-party
repos vendored side by side, and they freely use short, generic top-level module
names -- `model.py`, `config.py`, `utils.py` -- so two agents routinely ship
different modules under the same import name. Python caches modules globally in
``sys.modules``, so without intervention the first agent to import a given name
owns it for the rest of the run and every later agent silently gets the wrong
code (e.g. plant2's `dataset` being reused by plant1).

`PCLA.setup_agent` prepends each agent's own directory to ``sys.path`` before
loading it, which fixes *resolution* -- but only for names not already cached.
:func:`clear_agent_modules` drops the cached entries so the prepended path is
actually consulted.

Kept out of ``PCLA.py`` because it is policy data (a list of names that collide),
not loader logic: it grows every time an agent is added, and reads better as a
documented table than as a literal buried inside an import routine.
"""

import sys

__all__ = ['clear_agent_modules', 'AGENT_LOCAL_MODULES', 'AGENT_LOCAL_PREFIXES']


#: Top-level module names that more than one agent defines. Anything listed here
#: is evicted from ``sys.modules`` before each agent loads.
AGENT_LOCAL_MODULES = frozenset({
    # Generic names used by many of the vendored driving repos.
    'model', 'models', 'dataset', 'lit_module', 'plant_variables',
    'nav_planner', 'config', 'transfuser', 'transfuser_utils',
    'utils', 'util', 'data', 'gaussian_target',
    'planner', 'controller', 'map_agent', 'base_agent',
    'bev_planner', 'waypointer', 'lateral_controller',
    'longitudinal_controller', 'kinematic_bicycle_model',
    'carla_garage',

    # The VLA agents (`minddrive`, `orion`) each vendor their own *incompatible*
    # OpenMMLab fork under the top-level name `mmcv`, and both additionally ship
    # `adzoo` and `team_code`. `tt` meanwhile expects the real `mmcv-lite` from
    # site-packages. Whichever imported first would otherwise own these names for
    # the whole run, and the others would silently get the wrong package.
    'mmcv', 'adzoo', 'team_code', 'open_loop_training',
})

#: Submodule prefixes to evict. Clearing a package alone is not enough: entries
#: like ``mmcv.models`` or ``adzoo.orion`` stay cached and keep resolving into the
#: previous agent's tree even after the parent package is dropped.
AGENT_LOCAL_PREFIXES = (
    'models.',
    'util.',
    'carla_garage.',
    'birds_eye_view.',
    'mmcv.',
    'adzoo.',
    'team_code.',
    'open_loop_training.',
)


def clear_agent_modules(*extra_names):
    """Evict cached agent-local modules so the next agent re-imports its own.

    Call immediately before loading an agent, after its directory has been
    prepended to ``sys.path``.

    Args:
        *extra_names: additional module names to evict -- typically the dynamic
            module key of the agent being loaded, so a re-run of the same agent
            gets a fresh module object rather than the cached one.

    Returns:
        list[str]: the module names that were evicted (useful for debugging a
        suspected cross-agent contamination).
    """
    targets = AGENT_LOCAL_MODULES.union(extra_names)
    removed = []
    for key in list(sys.modules):
        if key in targets or key.startswith(AGENT_LOCAL_PREFIXES):
            del sys.modules[key]
            removed.append(key)
    return removed
