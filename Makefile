# Kaggle upload workflow for ARC Prize 2026 / ARC-AGI-3.
#
# One-time prerequisites:
#   pip install kaggle
#   place Kaggle API token at ~/.kaggle/kaggle.json
#
# Optional, for a private fork:
#   base64-encode a read-only GitHub deploy key and store it as the Kaggle
#   secret GITHUB_DEPLOY_KEY_ARC_PRIZE_2026.

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python
PUSH_NOSAVE := $(PYTHON) $(HERE)scripts/kaggle_push_nosave.py

id = $$($(PYTHON) -c "import json; print(json.load(open('$(HERE)$(1)kernel-metadata.json'))['id'])")

.PHONY: notebooks
notebooks:  ## regenerate all Kaggle notebooks + metadata
	$(PYTHON) $(HERE)build_notebooks.py

.PHONY: wheels wheels-run wheels-status
wheels: notebooks  ## quick-save the wheelhouse notebook; run it manually on Kaggle with internet + GPU
	$(PUSH_NOSAVE) $(HERE)wheels

wheels-run: notebooks  ## push the wheelhouse notebook and start a Kaggle run
	kaggle kernels push -p $(HERE)wheels

wheels-status:  ## check the latest wheelhouse notebook status
	kaggle kernels status $(call id,wheels/)

.PHONY: code code-run code-status
code: notebooks  ## quick-save the code/source-bundle notebook
	$(PUSH_NOSAVE) $(HERE)code

code-run: notebooks  ## push the code/source-bundle notebook and start a Kaggle run
	kaggle kernels push -p $(HERE)code

code-status:  ## check the latest code/source-bundle notebook status
	kaggle kernels status $(call id,code/)

.PHONY: submission submission-run submission-status
submission: notebooks  ## quick-save the scored submission notebook
	$(PUSH_NOSAVE) $(HERE).

submission-run: notebooks  ## push the scored submission notebook and start a Kaggle run
	kaggle kernels push -p $(HERE).

submission-status:  ## check the latest submission notebook status
	kaggle kernels status $(call id,)

.PHONY: status
status: wheels-status code-status submission-status  ## status of all generated Kaggle kernels

.PHONY: help
help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'
