import argparse
import random
import multiprocessing
import json
import tqdm
from simulator_sharded import ConversationSimulatorSharded

from simulator_full import ConversationSimulatorFull
from concurrent.futures import ThreadPoolExecutor
from utils_log import get_run_counts
from collections import Counter
from data_split import HELD_OUT, test_samples
import os

def run_simulation(todo):
    dataset_fn = todo["dataset_fn"]
    setting = todo["setting"]
    try:
        assistant_temp = todo.get("assistant_temperature", 1.0)
        user_temp = todo.get("user_temperature", 1.0)
        if setting == "full":
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
            conversation_simulator.run(verbose=args.verbose)
        elif setting == "sharded":
            conversation_simulator = ConversationSimulatorSharded(todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], user_model=todo["user_model"], mediator_model=todo["mediator_model"], assistant_temperature=assistant_temp, user_temperature=user_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
            conversation_simulator.run(verbose=args.verbose)
        else:
            conversation_simulator = ConversationSimulatorSharded(todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], user_model=todo["user_model"], mediator_model=todo["mediator_model"], assistant_temperature=assistant_temp, user_temperature=user_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
            conversation_simulator.run_mediator(verbose=args.verbose)
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        tqdm.tqdm.write(f"\033[91m [Error on {todo['sample']['task_id']}; {todo['assistant_model']}; {setting}]:\n{error_msg}\033[0m")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_file", type=str, default="data/sharded_instructions_600.json", help="Dataset file to use")

    parser.add_argument("--setting", choices=["ours", "sharded", "full"], default="ours", help="ours is the Mediator. sharded and full are the LiC baselines.")
    parser.add_argument("--n_runs", type=int, default=5, help="Runs per instance")
    parser.add_argument("--models", nargs="+", default=["deepseek-v3.2-thinking"], help="Assistant models")
    parser.add_argument("--mediator_model", type=str, default="deepseek-v3.2-thinking", help="Mediator model used by ours")
    parser.add_argument("--tasks", nargs="+", default=["math", "code", "database", "actions"], help="Tasks to run")
    parser.add_argument("--system_model", type=str, default="gpt-4o-mini", help="Model that judges turns and extracts answers")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini", help="Simulated user model")
    parser.add_argument("--N_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--log_folder", type=str, default=None, help="Defaults to logs/<setting>")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--assistant_temperature", type=float, default=1.0, help="Assistant temperature")
    parser.add_argument("--user_temperature", type=float, default=1.0, help="User temperature")

    args = parser.parse_args()
    if args.log_folder is None:
        args.log_folder = f"logs/{args.setting}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    

    dataset_fn = args.dataset_file
    if args.dataset_file.startswith(".\\"):
        dataset_fn = args.dataset_file[2:]
    dataset_fn = dataset_fn.replace("\\", "/")

    with open(dataset_fn, "r") as f:
        samples = json.load(f)

    samples = [d for d in samples if d["task"] in args.tasks]
    held = [d for d in samples if d["task_id"] in HELD_OUT]
    samples = test_samples(samples)
    print(f"Loaded {len(samples)} test samples ({len(held)} rule instances left out)")
    random.shuffle(samples)
    all_todos = []

    sharded_extra = f"-at{args.assistant_temperature}-ut{args.user_temperature}" if args.assistant_temperature != 1.0 or args.user_temperature != 1.0 else ""
    full_extra = f"-t{args.assistant_temperature}" if args.assistant_temperature != 1.0 else ""
    sharded_ct, full_ct = f"sharded{sharded_extra}", f"full{full_extra}"
    conv_type = full_ct if args.setting == "full" else sharded_ct

    all_tasks = list(set([sample["task"] for sample in samples]))
    for assistant_model in args.models:
        run_counts = Counter()
        for task in all_tasks:
            run_counts.update(get_run_counts(conv_type, task, assistant_model, dataset_fn, log_folder=args.log_folder))

        for sample in samples:
            remaining = args.n_runs - run_counts[sample["task_id"]]
            if remaining <= 0:
                continue
            all_todos += [{
                "sample": sample,
                "assistant_model": assistant_model,
                "mediator_model": args.mediator_model,
                "conv_type": conv_type,
                "setting": args.setting,
                "system_model": args.system_model,
                "user_model": args.user_model,
                "dataset_fn": dataset_fn,
                "assistant_temperature": args.assistant_temperature,
                "user_temperature": args.user_temperature,
            }] * remaining

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations ({args.setting})")
    print(Counter([todo["assistant_model"] for todo in all_todos]))
    print(Counter([todo["conv_type"] for todo in all_todos]))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(run_simulation, all_todos), total=len(all_todos), ncols=50))
