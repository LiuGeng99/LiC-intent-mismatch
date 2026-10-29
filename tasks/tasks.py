from tasks.database import TaskDatabase
from tasks.math import TaskMath
from tasks.code import TaskCode
from tasks.actions import TaskActions

def get_task(task_name, version=None):
    kwargs = {}
    if version is not None:
        kwargs["version"] = version

    if task_name.startswith("database"):
        return TaskDatabase(**kwargs)
    elif task_name == "code":
        return TaskCode(**kwargs)
    elif task_name == "math":
        return TaskMath(**kwargs)
    elif task_name.startswith("actions"):
        return TaskActions(**kwargs)
    else:
        raise ValueError(f"Task {task_name} not supported. This snapshot only includes code, database, actions, and math.")

if __name__ == "__main__":
    task = get_task("math")
    print(len(task.get_samples()))
