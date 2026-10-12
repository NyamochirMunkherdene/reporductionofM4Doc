# Test Project

## Overview

This repository is for experimenting with **Vary** and **M4Doc** in our HPC environment.

### Current Goal

* Verify that the environment is correctly configured.
* Confirm that the base Vary model can run successfully.
* Ensure all required dependencies and models are installed.

### Next Goal

* Reproduce the original **M4Doc** results.
* Verify that training and inference match the published implementation.
* Document any required modifications for the HPC environment.

## Project Structure

```text
test_project/
├── M4Doc/          # M4Doc framework
├── Vary/           # Vary implementation
├── Vary-toy/       # Lightweight Vary experiments
├── models/         # Downloaded pretrained models (not tracked by Git)
├── pyproject.toml
└── uv.lock
```

## Current Status

* [x] Repository created
* [x] GitHub repository configured
* [x] Environment setup completed
* [x] Base model installation completed
* [ ] Base inference successfully verified
* [ ] M4Doc reproduction
* [ ] Training experiments
* [ ] Performance evaluation

## Notes

* Large pretrained models are stored locally and are **not** included in this repository.
* Python virtual environments (`.venv`) are excluded from version control.
* This repository currently focuses on verifying the baseline implementation before starting research experiments.

## Future Work

* Reproduce M4Doc baseline results.
* Train and evaluate modified models.
* Document experimental settings and results.
* Compare baseline and proposed methods.

