# Standard Library
import json
import multiprocessing as mp
import os
from pathlib import Path

# Third Party
import pytest

# First Party
from smdebug.core.writer import FileWriter


def test_create_timeline_file(out_dir):
    timeline_writer = FileWriter(trial_dir=out_dir, step=0, worker=str(os.getpid()), wtype="trace")
    assert timeline_writer

    for i in range(1, 11):
        n = "event" + str(i)
        timeline_writer.write_trace_events(tensor_name="FileCreationTest", op_name=n, step_num=i)

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    for path in Path(out_dir + "/framework/pevents").rglob("*.json"):
        files.append(path)

    assert len(files) == 1

    with open(files[0]) as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict


def run(rank, timeline_writer):
    timeline_writer.write_trace_events(
        tensor_name="MultiProcessTest",
        op_name="event1",
        step_num=0,
        worker=os.getpid(),
        process_rank=rank,
    )
    timeline_writer.write_trace_events(
        tensor_name="MultiProcessTest", op_name="event2", step_num=1, worker=os.getpid()
    )
    timeline_writer.flush()


@pytest.mark.skip
def test_multiprocess_write(out_dir, monkeypatch):
    # monkeypatch.setenv(SM_PROFILER_FILE_PATH_ENV_STR, out_dir + "/test_timeline.json")
    timeline_writer = FileWriter(trial_dir=out_dir, step=0, worker=str(os.getpid()), wtype="trace")
    assert timeline_writer

    cpu_count = mp.cpu_count()

    processes = []
    for rank in range(cpu_count):
        p = mp.Process(target=run, args=(rank, timeline_writer))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    timeline_writer.close()

    files = []
    for path in Path(out_dir + "/framework/pevents").rglob("*.json"):
        files.append(path)

    assert len(files) == 1

    with open(files[0]) as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict


def test_duration_events(out_dir):
    timeline_writer = FileWriter(trial_dir=out_dir, step=0, worker=str(os.getpid()), wtype="trace")
    assert timeline_writer

    for i in range(1, 11):
        n = "event" + str(i)
        timeline_writer.write_trace_events(
            tensor_name="DurationEventTest", op_name=n, step_num=i, phase="B"
        )
        timeline_writer.write_trace_events(
            tensor_name="DurationEventTest", op_name=n, step_num=i, phase="E"
        )

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    for path in Path(out_dir + "/framework/pevents").rglob("*.json"):
        files.append(path)

    assert len(files) == 1

    with open(files[0]) as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict


@pytest.mark.parametrize("policy", ["file_size", "file_interval"])
def test_rotation_policy(out_dir, monkeypatch, policy):
    if policy == "file_size":
        monkeypatch.setenv("ENV_MAX_FILE_SIZE", "350")
    elif policy == "file_interval":
        monkeypatch.setenv("ENV_CLOSE_FILE_INTERVAL", "1")
    timeline_writer = FileWriter(
        trial_dir=out_dir, step=0, worker=str(os.getpid()), wtype="trace", flush_secs=1
    )
    assert timeline_writer

    for i in range(1, 11):
        n = "event" + str(i)
        timeline_writer.write_trace_events(tensor_name="FileCreationTest", op_name=n, step_num=i)

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    for path in Path(out_dir + "/framework/pevents").rglob("*.json"):
        files.append(path)

    assert len(files) == 1

    with open(files[0]) as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict
