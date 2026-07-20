"""Compatibility shims for running MindDrive inside PCLA's shared environment.

Imported first from this package's ``__init__`` so the fixes land before any
third-party module that depends on them.

PCLA pins matplotlib 3.5.3 (shared by ~20 other agents), while the nuscenes-devkit
build MindDrive needs targets matplotlib >= 3.6. Rather than bump matplotlib for the
whole environment -- which would risk every other agent -- we reconcile the one
incompatibility that actually bites.
"""


def _alias_seaborn_styles():
    """Map matplotlib >= 3.6 seaborn style names onto their 3.5 equivalents.

    matplotlib 3.6 renamed the bundled seaborn styles (``seaborn-whitegrid`` ->
    ``seaborn-v0_8-whitegrid``). nuscenes-devkit's ``map_expansion.map_api`` calls
    ``plt.style.use('seaborn-v0_8-whitegrid')`` at import time, which raises
    ``OSError`` on 3.5 and makes the whole dataset module unimportable. Registering
    the new names as aliases of the old ones is a no-op on matplotlib >= 3.6.
    """
    try:
        import matplotlib.style as mstyle
    except ImportError:  # pragma: no cover - matplotlib always present in PCLA
        return

    library = getattr(mstyle, 'library', None)
    if not library:
        return

    added = False
    for old_name in list(library):
        if not old_name.startswith('seaborn-') or old_name.startswith('seaborn-v0_8'):
            continue
        new_name = old_name.replace('seaborn-', 'seaborn-v0_8-', 1)
        if new_name not in library:
            library[new_name] = library[old_name]
            added = True

    if added and hasattr(mstyle, 'available'):
        # `available` is a module-level list mirroring the library keys.
        mstyle.available[:] = sorted(library.keys())


_alias_seaborn_styles()
