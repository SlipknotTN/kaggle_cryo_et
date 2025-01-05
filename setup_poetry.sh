#!/usr/bin/env bash
# Force poetry to use the conda environment, not isolated from other projects
poetry config virtualenvs.path /home/michele/anaconda3/envs/pytorch_poetry
poetry config virtualenvs.create false