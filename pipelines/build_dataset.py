"""Build an immutable training dataset from curated offline data.

This pipeline will validate source data and annotations, create or verify
deterministic splits, record provenance and content hashes, write a dataset
manifest, and publish the resulting version as a W&B dataset artifact.

Data collection, generation, labeling, and curation happen before this
pipeline. Production observations are only one optional source of candidate
data and receive no automatic path into a dataset version.
"""
