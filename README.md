# CSE 151B Competition

**Name, PID:** Michael Sun, A19164414

**Group:** MWS

Run instructions:
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