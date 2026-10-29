
RULE_IDS = {
    "code": [
        "sharded-HumanEval/150",
        "sharded-HumanEval/88",
        "sharded-HumanEval/122",
        "sharded-livecodebench/2786",
        "sharded-livecodebench/2845",
    ],
    "database": [
        "sharded-spider-val-457-medium",
        "sharded-spider-val-257-hard",
        "sharded-spider-val-794-extra",
        "sharded-spider-val-611-medium",
        "sharded-spider-val-571-hard",
    ],
    "actions": [
        "sharded-BFCL/parallel_144",
        "sharded-BFCL/parallel_199",
        "sharded-BFCL/parallel_180",
        "sharded-BFCL/parallel_195",
        "sharded-BFCL/parallel_107",
    ],
    "math": [
        "sharded-GSM8K/1027",
        "sharded-GSM8K/968",
        "sharded-GSM8K/201",
        "sharded-GSM8K/1019",
        "sharded-GSM8K/825",
    ],
}

HELD_OUT = {task_id for ids in RULE_IDS.values() for task_id in ids}

def test_samples(samples):
    return [sample for sample in samples if sample["task_id"] not in HELD_OUT]
