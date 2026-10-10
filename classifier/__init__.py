"""
Vision Language Model adapters.

`nemotron.py` (`NemotronVisionModel`) is the one concrete adapter, wrapping
NVIDIA Build's Nemotron 3 Nano Omni model. It exposes two capabilities used
by the two orchestrators: `select_best_candidate()` (comparative multi-image
ranking, used by `selection_stage.py`) and `verify_product_match()`
(single-image product-match verification, used by
`retailer_gallery_stage.py`). Both are specific enough to this adapter that
there is deliberately no abstract vision port in `interfaces.py` - callers
depend on `NemotronVisionModel` concretely. See `interfaces.py`'s module
docstring for the reasoning, and for when reintroducing a port would be
justified.
"""
