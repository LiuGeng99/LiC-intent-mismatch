# Intent Mismatch

Reproduction code for the Mediator–Assistant pipeline on the [Lost in Conversation](https://arxiv.org/abs/2505.06120) benchmark.

## Setup

```sh
pip install -r requirements.txt
export OPENAI_API_KEY=...
```

Set `OPENAI_BASE_URL` when using a non-OpenAI endpoint.

## Data

Sharded instructions are included at `data/sharded_instructions_600.json`.

The Database task additionally requires the Spider test-suite SQLite files (under 5 GB), which are not distributed here. Download them from [Google Drive](https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view?usp=sharing) ([test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)). After unpacking, copy each database directory from `database/` into `data/spider/databases/`. The path `data/spider/databases/concert_singer/concert_singer.sqlite` should then exist.

## Evaluation

```sh
python run_simulations.py --models gpt-4o-mini --tasks math --n_runs 5
python score_results.py
```

Logs are written to `logs/ours/`. `score_results.py` reports average performance (P) and reliability (R).

Single-turn (Full) and multi-turn (Sharded) baselines:

```sh
python run_simulations.py --setting full --models gpt-4o-mini --tasks math --n_runs 5
python run_simulations.py --setting sharded --models gpt-4o-mini --tasks math --n_runs 5
```

## Experience extraction

To extract a new rule set, provide your own examples to `run_refiner.py`. Each example is an object with a dialogue (`conversation`) and the corresponding fully specified request (`ground_truth`):

```json
[
  {
    "conversation": "[User]: ...\n[AI]: ...",
    "ground_truth": "..."
  }
]
```

```sh
python run_refiner.py --examples examples.json
```

Place the resulting rules under `[Important Rules]` in `prompts/mediator.txt` before running evaluation with that rule set.

## License

Code and sharded instructions derived from [Lost in Conversation](https://github.com/microsoft/lost_in_conversation) are included under the MIT License. Copyright (c) Microsoft Corporation. See `LICENSE`.
