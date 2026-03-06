# Refactor Plan: `femr` -> `ora`, Path Anonymization, and Script Comment Cleanup

## Goal
Perform a repository-wide refactor with three outcomes:
1. Rename package/module usage from `femr` to `ora` across code, docs, and configuration.
2. Anonymize real absolute paths in README files by replacing them with general placeholders.
3. Remove end-of-file `.py` script-running comment blocks (for example blocks containing `export ...` and `python ...` commands).

## Scope
- In scope:
  - `src/` package directory and all Python imports/usages.
  - `pyproject.toml` and related packaging entry points.
  - All README files in the repo (currently `README.md` and `src/femr/omop_meds_tutorial/evaluation/regression/extract_cohort_cumc/readme.md`).
  - End-of-file command/script comment blocks in `.py` files.
- Out of scope:
  - Binary artifacts (for example `.whl`) unless explicitly requested.
  - Behavioral/model logic changes unrelated to naming/anonymization/cleanup.

## Execution Plan

### Phase 1: Rename package references (`femr` -> `ora`)
1. Rename package directory:
   - `src/femr` -> `src/ora`.
2. Update packaging metadata:
   - `pyproject.toml` package name and version file path.
   - Console script entry points currently targeting `femr...`.
3. Update source imports and module references:
   - `import femr...`, `from femr...`, type hints like `femr.x.y`.
   - `python -m femr...` command strings in scripts/docs.
4. Update remaining textual package-name mentions in tracked text files where relevant.

### Phase 2: Anonymize absolute paths in README files
1. Replace real absolute paths with placeholders in README files only.
2. Use consistent placeholder naming style such as:
   - `$Path_to_meds_reader`
   - `$Path_to_pretraining_data`
   - `$Path_to_model_checkpoint`
   - `$Path_to_ontology`
   - `$Path_to_main_split_csv`
   - `$Path_to_output_dir`
   - `$Path_to_hf_cache`
3. Keep command examples runnable as templates (paths replaced, argument structure preserved).

### Phase 3: Remove trailing `.py` script command blocks
1. Identify `.py` files with end-of-file blocks used as run scripts, including patterns like:
   - `export CUDA_VISIBLE_DEVICES=...`
   - `python ... \`
2. Remove these trailing script blocks while preserving:
   - Real module docstrings.
   - Functional comments needed for code understanding.
3. Recheck files to ensure removal is only applied to script-like footer blocks.

## Validation Checklist
1. Package rename sanity:
   - `rg -n "\bfemr\b"` should return no live package/import/config references intended to be renamed.
2. README anonymization:
   - `rg -n "(/Users/|/user/|/data/|/home/|/scratch/|/shared/)" README.md src/**/readme.md` should return no matches.
3. Script footer cleanup:
   - Spot-check changed `.py` files for removed footer command blocks and intact code.
4. Basic import/build consistency:
   - `python -m compileall src/ora`
   - `python -c "import ora; print(ora.__name__)"`
5. Optional full checks (if requested):
   - `pytest -q`
   - `pre-commit run --all-files`

## Risks and Mitigations
- Risk: Rename may miss indirect references (string literals, CLI snippets, or config pointers).
  - Mitigation: Validate with repo-wide search and import/compile checks.
- Risk: Over-replacing plain text `femr` might affect historical attribution text.
  - Mitigation: Review doc replacements contextually; preserve meaning where needed.
- Risk: Removing comment blocks may accidentally delete useful notes.
  - Mitigation: Only delete clear script-runner footer blocks, not explanatory inline comments.

## Deliverables
- Refactored package/module naming to `ora`.
- README files with anonymized path placeholders.
- `.py` files cleaned from end-of-file script command blocks.
- Validation results summary plus a changed-files list.
