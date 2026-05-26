.PHONY: help install figures train-hippie train-physmap train-nemo train-wfrf clean

help:
	@echo "HIPPIE benchmarking — minimal release"
	@echo ""
	@echo "Setup:"
	@echo "  make install           Create conda env 'hippie' and install package"
	@echo ""
	@echo "Quick reproduction (cached results):"
	@echo "  make figures           Regenerate every figure from cached predictions"
	@echo ""
	@echo "Full reproduction (training, GPU recommended):"
	@echo "  make train-hippie      Run HIPPIE on every dataset × 5 CV folds"
	@echo "  make train-physmap     Run PhysMAP (R) on every dataset"
	@echo "  make train-nemo        Run NEMO on every dataset with the 3D ACG"
	@echo "  make train-wfrf        Run handcrafted-feature RandomForest baseline"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             Remove caches and generated artefacts"

install:
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n hippie pip install -e ".[dev]"
	@echo ""
	@echo "Environment 'hippie' is ready."
	@echo "Activate with: conda activate hippie"

figures:
	bash scripts/run_all_figures.sh

train-hippie:
	@echo "HIPPIE training is per-dataset × per-fold. See:"
	@echo "  docs/REPRODUCING.md   for the full dataset × fold matrix"
	@echo "  python scripts/train_multimodal_transductive.py --help"
	@echo "  python scripts/train_multimodal_holdout.py --help"
	@echo "  python scripts/cross_dataset_script.py --help"

train-physmap:
	bash comparison_methods/physmap/run_physmap.sh
	bash comparison_methods/physmap/run_physmap_holdout.sh
	bash comparison_methods/physmap/run_physmap_crossdataset.sh

train-nemo:
	bash comparison_methods/nemo/scripts/run_nemo_benchmark.sh

train-wfrf:
	@echo "WF-RF runs per dataset × per fold. See:"
	@echo "  python comparison_methods/wf-rf/wf_rf_benchmark_evaluation.py --help"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
