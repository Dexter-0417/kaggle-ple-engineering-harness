# Physical-Engineering's Last Exam — competition entry

This is a separate harness repository based on the full MCP-enabled custom-openai sample from Google's PLE public repository (Apache 2.0).

The entry targets `google-cloud/claude-opus-5-5` for the official competition evaluator. A few public development tasks have been evaluated experimentally through the user's local Claude Code Opus subscription. Those are genuine public-dev checks-grader results, but the local CLI proxy is **not** the competition's official Google model gateway. We have no 29-task total score. Do not interpret schema checks, offline tests, Docker builds or single-task results as a leaderboard score.

Rules and reference: https://www.kaggle.com/competitions/physical-engineerings-last-exam ; https://ple-bench.com/competition

## Submission items

The Kaggle hackathon uses one team Writeup (up to 2,000 words), GitHub source URL, a full 40-character Git commit identifier, a cover image and three figures: system architecture, score against cost, and dev task failure breakdown. The repository should include this harness and dev artifacts produced by the competition evaluator; commit and score only actual `eval/run.py` outputs. Public repository access or appropriate official app permissions are required.

The dev dataset is gated on Hugging Face for download. Local model calls ordinarily need an authenticated model proxy and may incur Vertex AI charges. This team has separately tested an authorized subscription-based Claude Code CLI bridge for experimental local tasks, but its cost accounting and model behavior must not be equated with the production gateway. Never put credentials in this repository.

## Development status

The official kit's schema validation and offline Docker builds have passed. The public source and free GitHub Actions CI are at https://github.com/Dexter-0417/kaggle-ple-engineering-harness. Local public development results: data-center-site-model 1.000/1.000; data-hall-mep-routing 0.000/1.000 in both text-only and initial vision-enabled pilots. These are single experimental CLI-based runs, not a 29-task mean. Production-model `kit check` from the container still needs authorized model connectivity. The stage-1 Kaggle Writeup has not been submitted.

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
