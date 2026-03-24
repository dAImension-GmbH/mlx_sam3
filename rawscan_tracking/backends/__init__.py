from .local import LocalTrackerBackend

__all__ = ["LocalTrackerBackend", "SAM3Refiner", "SAM3RefinementTrackerBackend"]


def __getattr__(name: str):
    if name in {"SAM3Refiner", "SAM3RefinementTrackerBackend"}:
        from .sam3_refine import SAM3Refiner, SAM3RefinementTrackerBackend

        return {
            "SAM3Refiner": SAM3Refiner,
            "SAM3RefinementTrackerBackend": SAM3RefinementTrackerBackend,
        }[name]
    raise AttributeError(name)
