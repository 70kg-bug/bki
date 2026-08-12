"""Serving-importable logic. Nothing here imports a stage.

bands, charlson, explain and grounding additionally read no config and do no I/O:
they take their policy as an argument, so serving can import them without the
pipeline. scoring, generate and xgb_math read config but still never import a stage.
"""
