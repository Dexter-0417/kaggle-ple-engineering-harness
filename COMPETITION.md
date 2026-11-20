# Physical-Engineering's Last Exam — competition entry

This is a separate harness repository based on the full MCP-enabled custom-openai sample from Google's PLE public repository (Apache 2.0).

The entry targets `google-cloud/claude-opus-5-5`. No paid model execution or dev task evaluation has been performed. Do not interpret the local schema validation or Docker build as a benchmark score.

Rules and reference: https://www.kaggle.com/competitions/physical-engineerings-last-exam ; https://ple-bench.com/competition

## Submission items

The Kaggle hackathon uses one team Writeup (up to 2,000 words), GitHub source URL, a full 40-character Git commit identifier, a cover image and three figures: system architecture, score against cost, and dev task failure breakdown. The repository should include this harness and dev artifacts produced by the competition evaluator; commit and score only actual `eval/run.py` outputs. Public repository access or appropriate official app permissions are required.

The local dev dataset is gated on Hugging Face for download, even though Kaggle's Writeup entry form itself does not ask for Hugging Face authorization. Model calls for local tests need an authenticated model proxy and may incur Vertex AI charges. Never put credentials in this repository.

## Development status

`python -m kit validate` passed on the official kit. Docker builder completed and bundled the harness (115 MiB). `kit check` stopped at the missing model proxy environment variables. Kaggle Team page saved successfully and confirmed a team leader. No score yet, and no GitHub remote configured.

## Offline test scope

The test suite uses mocked MCP servers, synthetic output files and an in-memory
PLE evaluator protocol. No hidden or public benchmark task solution is
hardcoded in tests, and none of these tests represents an official benchmark
score. The output verification tool does not inspect grader code or reference
answers.

## Engineering constraint checker guardrails

The harness can parse restricted numeric expressions and compare them with
numerical limits quoted verbatim from the current evaluator instruction.
This includes strict source-case and unit-symbol matching to avoid confusing
MPa/mPa, MW/mW or composite units, plus conservative comparison-word checks.
Outputs are marked PASS, FAIL or UNVERIFIED. These checks deliberately cannot
assert full compliance, determine an authoritative formula, or reconstruct
source values from attachments; the engineer/model must independently verify
those. No benchmark-specific threshold or solution is hard-coded. All tests use
synthetic scenarios and do not inspect task graders.
