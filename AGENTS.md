# Repository Guidelines

## Project Structure & Module Organization
- `soni_translate/` holds the core pipeline (segmentation, translation, TTS, post-processing) and shared utilities.
- `lib/` contains model and inference helpers (e.g., `infer_pack`, `rmvpe`).
- `app_rvc.py` is the main Gradio UI entry point; `voice_main.py` and `vci_pipeline.py` are auxiliary scripts.
- `assets/` stores static media (logo); `docs/` contains install guides; `mdx_models/` stores metadata.
- `SoniTranslate_Colab*.ipynb` are Colab notebooks for hosted runs.

## Build, Test, and Development Commands
- Create the conda env: `conda create -n sonitr python=3.10 -y`
- Install dependencies:
  - `pip install -r requirements_base.txt`
  - `pip install -r requirements_extra.txt`
  - `pip install onnxruntime-gpu`
- Run locally: `python app_rvc.py` (starts the Gradio UI at `http://127.0.0.1:7860`).
- Optional TTS add-ons: `pip install -r requirements_xtts.txt` then `pip install TTS==0.21.1 --no-deps`.
- Dev container support lives in `.devcontainer/` (`Dockerfile`, `postCreate.sh`).

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard snake_case for functions/variables, CamelCase for classes.
- Keep UI wiring in `app_rvc.py`, and prefer placing new pipeline logic in `soni_translate/`.
- There is no enforced formatter or linter; follow existing patterns in neighboring files.

## Testing Guidelines
- No automated test suite is configured in this repo.
- For changes, run a manual smoke test by launching `python app_rvc.py` and processing a short sample clip.
- If you add tests, place them under `tests/` and document how to run them in the PR.

## Commit & Pull Request Guidelines
- Recent history uses Conventional Commits (e.g., `feat: ...`, `fix: ...`); follow that pattern.
- PRs should include a short summary, how you tested (CPU/GPU), and UI screenshots when the Gradio flow changes.
- Link related issues and note any model/weight changes or new dependencies.

## Configuration & Security Tips
- Do not commit secrets. Use environment variables like `YOUR_HF_TOKEN` and `OPENAI_API_KEY`.
- Some models require Hugging Face license acceptance; call that out in PR notes when relevant.
