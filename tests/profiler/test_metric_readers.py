# First Party
from smdebug.profiler.MetricsReader import LocalMetricsReader, S3MetricsReader
from smdebug.profiler.utils import TimeUnits


def test_S3MetricsReader():
    bucket_name = "tornasole-dev"
    tt = S3MetricsReader(bucket_name)
    events = tt.get_events(1589930980, 1589930995, unit=TimeUnits.SECONDS)
    print(f"Number of events {len(events)}")


def test_LocalMetricsReader(tracefolder="./tests/profiler/test_traces"):
    lt = LocalMetricsReader(tracefolder)
    events = lt.get_events(1589930980, 1589930995, unit=TimeUnits.SECONDS)
    print(f"Number of events {len(events)}")
    assert len(events) == 4


def test_LocalMetricsReader_Model_timeline(tracefolder="./tests/profiler/model_timeline_traces"):
    lt = LocalMetricsReader(tracefolder)
    events = lt.get_events(1590104140534059, 1590104153293665)
    print(f"Number of events {len(events)}")
    assert len(events) == 125
