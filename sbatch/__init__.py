"""Batch-job helper scripts for launching and supporting long training runs.

This directory mainly contains executable helpers that are started from Slurm batch scripts.
Adding a package marker keeps local imports and static analysis consistent in tests without
changing how the scripts are invoked from the command line.
"""
