#!/bin/bash
#PBS -l select=1:ncpus=5:mem=20gb:ngpus=1:accelerator_model=a100
#PBS -l walltime=1:59:00
#PBS -A "B-cell-aging"

set -e  

module load Python/3.12.3

export PATH="$HOME/.local/bin:$PATH"

export POETRY_PYPI_MIRROR_URL=http://pypi.repo.test.hhu.de/simple/

cd /gpfs/project/nirec101/transformer_project

poetry install

poetry run python transformer_project/run/train_model.py

