# Standard Library
import time

# First Party
from smdebug.profiler.profiler_constants import PROFILER_NUM_STEPS_DEFAULT


class RotationPolicy:
    """Configuration corresponding to rotation policy of trace event files.
    """

    def __init__(self, file_max_size, file_close_interval):
        self.file_max_size = file_max_size
        self.file_close_interval = file_close_interval


class TraceFile:
    """Configuration corresponding to trace event files.
    """

    def __init__(self, file_max_size, file_close_interval, file_open_fail_threshold):
        self.rotation_policy = RotationPolicy(file_max_size, file_close_interval)
        self.file_open_fail_threshold = file_open_fail_threshold


class ProfileRange:
    """Configuration corresponding to the detailed profiling config.
    Assigns each field from the dictionary, (or `None` if it doesn't exist)
    """

    def __init__(self, profile_range):
        # step range
        self.start_step = profile_range.get("StartStep")
        self.num_steps = profile_range.get("NumSteps")
        self.end_step = None

        # time range
        self.start_time = profile_range.get("StartTime")
        self.duration = profile_range.get("Duration")
        self.end_time = None

    def has_step_range(self):
        return self.start_step or self.num_steps

    def has_time_range(self):
        return self.start_time or self.duration

    def can_start_detailed_profiling(self, current_step, current_time=time.time()):
        """Determine whether the values from the config are valid for detailed profiling.
        """
        if self.has_step_range():
            if not self.start_step:
                self.start_step = current_step
            if not self.num_steps:
                self.num_steps = PROFILER_NUM_STEPS_DEFAULT
            if not self.end_step:
                self.end_step = self.start_step + self.num_steps
            return self.start_step <= current_step < self.end_step
        elif self.has_time_range():
            if not self.start_time:
                self.start_time = current_time
            if self.duration:
                if not self.end_time:
                    self.end_time = self.start_time + self.duration
                return self.start_time <= current_time < self.end_time
            else:
                if self.start_time <= current_time:
                    if not self.end_step:
                        self.end_step = current_step + 1
                    return current_step < self.end_step
        return False


class ProfilerConfig:
    """Overall profiler configuration
    """

    def __init__(
        self,
        local_path,
        file_max_size,
        file_close_interval,
        file_open_fail_threshold,
        profile_range,
    ):
        """
        :param local_path: path where profiler events have to be saved.
        :param file_max_size: Max size a trace file can be, before being rotated.
        :param file_close_interval: Interval in seconds from the last close, before being rotated.
        :param file_open_fail_threshold: Number of times to attempt to open a trace fail before marking the writer as unhealthy.
        :param profile_range Dictionary holding the detailed profiling config.
        """
        self.local_path = local_path
        self.trace_file = TraceFile(file_max_size, file_close_interval, file_open_fail_threshold)
        self.profile_range = ProfileRange(profile_range)
