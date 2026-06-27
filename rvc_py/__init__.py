"""rvc-py: Modern Python library for Retrieval-based Voice Conversion."""

__version__ = "0.1.0"
__all__ = ["RVCModel"]

# Ленивый импорт — torch не грузится при import rvc_py
def __getattr__(name):
    if name == "RVCModel":
        from rvc_py.rvc_model import RVCModel
        return RVCModel
    raise AttributeError(f"module 'rvc_py' has no attribute {name!r}")