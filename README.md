# Caliper

A generic, metadata-driven LLM evaluation framework.

Caliper runs the Cartesian product of `(prompt × model × test_case × iteration)`,
sends each output to a configurable judge, and writes structured traces +
per-dimension scores to [Langfuse](https://langfuse.com) — so the
"which prompt/model combo wins?" question has an out-of-the-box answer.

## Status

Early development. Not yet ready for general use.

## Philosophy

- **Test cases own the rubric.** The judge is a generic executor; what
  it's judging *against* lives on the test case's metadata. One framework,
  N eval types (code review, agent tool-call verification, agent outcome
  correctness, …) — selected by a single `eval_type` discriminator.
- **Configs in folders, framework in code.** Adding a new eval type or a
  new test case means dropping files into a folder — not editing the
  framework. The framework is the runner; the configs are the work.
- **OSS dependencies all the way down.** [Langfuse](https://langfuse.com)
  for trace + score storage. [LiteLLM](https://github.com/BerriAI/litellm)
  for multi-vendor LLM calls. Everything runs in your own Docker.

## Quickstart

(Pending — coming with Milestone 4.)

```bash
docker compose up -d
cp .env.example .env  # fill in API keys
python -m caliper.eval_runner examples/code_review/eval_config.yaml
```

## License

[MIT](LICENSE) — use it however you like.
