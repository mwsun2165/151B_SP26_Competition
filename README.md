# CSE 151B Competition

**Name, PID:** Michael Sun, A19164414

**Group:** MWS

**GPU Type Used:** Ran on Runpod with the `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` template. Used `H100 SXM` gpu. Inference on full private dataset takes around `3` hours.

**Run instructions:**
1. Setup virtual environment (optional with uv) and install required libraries:
```
!wget -qO- https://astral.sh/uv/install.sh | sh

!uv venv .venv --seed

!uv pip install -r requirements.txt

!uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git --no-build-isolation
```
2. Place private dataset .jsonl in `data/private.jsonl`
3. Call `run_inference` from `main.py`
4. Submission .csv will be located in cwd, named `submission.csv`

Note: modify `DATA_PATH` and `SUBMISSION_PATH` in `main.py` to change the data file path and the submission file if needed.