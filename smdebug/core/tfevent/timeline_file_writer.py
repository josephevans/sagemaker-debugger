# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Writes trace events to disk in trial dir or user-specified dir."""

# Standard Library
import collections
import json
import os
import threading
import time
from datetime import datetime

# Third Party
import six

# First Party
from smdebug.core.access_layer.file import SMDEBUG_TEMP_PATH_SUFFIX
from smdebug.core.locations import TraceFileLocation
from smdebug.core.logger import get_logger
from smdebug.core.utils import ensure_dir, get_node_id
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS, PYTHONTIMELINE_SUFFIX

logger = get_logger()


def _get_sentinel_event(base_start_time):
    """Generate a sentinel trace event for terminating worker."""
    return TimelineRecord(timestamp=time.time(), base_start_time=base_start_time)


"""
TimelineRecord represents one trace event that ill be written into a trace event JSON file.
"""


class TimelineRecord:
    def __init__(
        self,
        timestamp,
        base_start_time,
        training_phase="",
        phase="X",
        operator_name="",
        args=None,
        duration=0,
    ):
        """
        :param timestamp: Mandatory field. Absolute start_time for the event in seconds.
        :param training_phase: strings like, train_step_time , test_step_time etc
        :param phase: Trace event phases. Example "X", "B", "E", "M"
        :param operator_name: more details about phase like whether dataset or iterator
        :param args: other information to be added as args
        :param duration: any duration manually computed (in seconds)
        """
        self.training_phase = training_phase
        self.phase = phase
        self.op_name = operator_name
        self.args = args
        self.base_start_time = base_start_time
        abs_ts_micros = int(timestamp * CONVERT_TO_MICROSECS)
        self.rel_ts_micros = abs_ts_micros - self.base_start_time
        self.duration = (
            duration
            if duration is not None
            else int(round(time.time() * CONVERT_TO_MICROSECS) - abs_ts_micros)
        )
        self.event_end_ts_micros = abs_ts_micros + self.duration
        self.pid = 0
        self.tid = 0

    def to_json(self):
        json_dict = {
            "name": self.op_name,
            "pid": self.pid,
            "ph": self.phase,
            "ts": self.rel_ts_micros,
        }

        # handle Instant event
        if self.phase == "i":
            if self.args:
                # Instant events have a field unique to them called scope.
                # scope can be "g" - global, "p" - process, "t" - thread.
                # parsing this value that is being passed as args.
                s = self.args["s"] if "s" in self.args else "t"
                json_dict.update({"s": s})
                if "s" in self.args:
                    self.args.pop("s")
        elif self.phase == "X":
            json_dict.update({"dur": self.duration})

        if self.args:
            json_dict["args"] = self.args

        return json.dumps(json_dict)


class TimelineFileWriter:
    """This class is adapted from EventFileWriter in Tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py
    Writes TimelineRecord to a JSON trace file.
    The `TimelineFileWriter` class creates a timeline JSON file in the specified directory,
    and asynchronously writes TimelineRecord to the file.
    """

    def __init__(self, profiler_config_parser, max_queue=100, suffix=PYTHONTIMELINE_SUFFIX):
        """Creates a `TimelineFileWriter` and a trace event file to write to.
        This event file will contain TimelineRecord as JSON strings, which are written to
        disk via the write_record method.
        If the profiler is not enabled, trace events will not be written to the file.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        """
        self.start_time_since_epoch_in_micros = int(round(time.time() * CONVERT_TO_MICROSECS))
        self._profiler_config_parser = profiler_config_parser
        self._max_queue = max_queue
        self._flush_secs = 1
        self._flush_complete = threading.Event()
        self._flush_sentinel = object()
        self._close_sentinel = object()
        self._sentinel_event = _get_sentinel_event(self.start_time_since_epoch_in_micros)
        self._initialize(suffix)
        self._closed = False

    def _initialize(self, suffix):
        """Initializes or re-initializes the queue and writer thread.
        The EventsWriter itself does not need to be re-initialized explicitly,
        because it will auto-initialize itself if used after being closed.
        """
        self._event_queue = CloseableQueue(self._max_queue)
        self._worker = _TimelineLoggerThread(
            queue=self._event_queue,
            flush_secs=self._flush_secs,
            flush_complete=self._flush_complete,
            flush_sentinel=self._flush_sentinel,
            close_sentinel=self._close_sentinel,
            base_start_time_in_us=self.start_time_since_epoch_in_micros,
            profiler_config_parser=self._profiler_config_parser,
            suffix=suffix,
        )
        self._worker.start()

    def _update_base_start_time(self, base_start_time_in_us):
        """
        Some trace files such as the Horovod trace file may start before this timeline
        writer is initialized. In such case, use this function to update the start time
        since epoch in micros.
        """
        if base_start_time_in_us != self.start_time_since_epoch_in_micros:
            self.start_time_since_epoch_in_micros = base_start_time_in_us
            self._worker._update_base_start_time(base_start_time_in_us)

    def write_trace_events(
        self, timestamp, training_phase="", op_name="", phase="X", duration=0, **kwargs
    ):
        """
        Creates TimelineRecord from the details passed as parameters, and enqueues an event for write.
        :param timestamp:start_time for the event (in seconds)
        :param training_phase: strings like, data_iteration, forward, backward, operations etc
        :param op_name: more details about phase like whether dataset or iterator
        :param phase: phase of trace event. default is 'X'
        :param duration: any duration manually computed (in seconds)
        :param kwargs: other params. can be process id and thread id
        """
        if not self._worker._healthy or not self._profiler_config_parser.profiling_enabled:
            return
        duration_in_us = int(duration * CONVERT_TO_MICROSECS)  # convert to micro seconds
        args = {**kwargs}
        event = TimelineRecord(
            training_phase=training_phase,
            operator_name=op_name,
            phase=phase,
            timestamp=timestamp,
            args=args,
            duration=duration_in_us,
            base_start_time=self.start_time_since_epoch_in_micros,
        )
        self._try_put(event)

    def _try_put(self, item):
        """Attempts to enqueue an item to the event queue.
        If the queue is closed, this will close the EventFileWriter and reraise the
        exception that caused the queue closure, if one exists.
        Args:
          item: the item to enqueue
        """
        try:
            self._event_queue.put(item)
        except QueueClosedError:
            self._internal_close()
            if self._worker.failure_exc_info:
                six.reraise(*self._worker.failure_exc_info)  # pylint: disable=no-value-for-parameter

    def write_event(self, event):
        """Adds a trace event to the JSON file."""
        self._try_put(event)

    def flush(self):
        """Flushes the trace event file to disk.
        Call this method to make sure that all pending events have been written to disk.
        """
        if not self._closed:
            # Request a flush operation by enqueing a sentinel and then waiting for
            # the writer thread to mark the flush as complete.
            self._flush_complete.clear()
            self._try_put(self._flush_sentinel)
            self._flush_complete.wait()
            if self._worker.failure_exc_info:
              self._internal_close()
              six.reraise(*self._worker.failure_exc_info)  # pylint: disable=no-value-for-parameter

    def close(self):
        """Flushes the trace event file to disk and close the file.
        """
        if not self._closed:
            self.flush()
            self._try_put(self._close_sentinel)
            self._internal_close()

    def _internal_close(self):
        """Flushes the trace event file to disk and close the file.
        """
        self._closed = True
        self._worker.join()
        self._worker.close()


class _TimelineLoggerThread(threading.Thread):
    """Thread that logs events. Copied from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py#L133"""

    def __init__(
        self,
        queue,
        flush_secs,
        flush_complete,
        flush_sentinel,
        close_sentinel,
        base_start_time_in_us,
        profiler_config_parser,
        verbose=False,
        suffix=PYTHONTIMELINE_SUFFIX,
    ):
        """Creates a _TimelineLoggerThread."""
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._flush_secs = flush_secs
        self._next_event_flush_time = 0
        self._flush_complete = flush_complete
        self._flush_sentinel = flush_sentinel
        self._close_sentinel = close_sentinel
        self._num_outstanding_events = 0
        self._writer = None
        self.verbose = verbose
        # This is a master dictionary that keeps the track of training_phase to pid globally. The dictionary will not
        # get
        # reset for a new file. This ensures that we keep unique pid for the training phases.
        self.training_phase_to_pid = collections.defaultdict(int)
        # This table keeps track of training_phase to pid for a given file. It will be reset for every new file. For
        # a given training phase, if we don't find entry in this table, we will write metaevent for that training
        # phase.
        self.tensor_table = collections.defaultdict(int)
        self.continuous_fail_count = 0
        self.is_first = True
        self._update_base_start_time(base_start_time_in_us)
        self._healthy = True
        self._profiler_config_parser = profiler_config_parser
        self.node_id = get_node_id()
        self.suffix = suffix
        self.failure_exc_info = ()

    def _update_base_start_time(self, base_start_time_in_us):
        """
        Some trace files such as the Horovod trace file may start before this timeline
        writer is initialized. In such case, use this function to update the start time
        since epoch in micros.
        """
        self.last_event_end_time_in_us = int(round(base_start_time_in_us))
        self.last_file_close_time_in_us = self.last_event_end_time_in_us
        self.cur_hour = datetime.utcfromtimestamp(
            self.last_file_close_time_in_us / CONVERT_TO_MICROSECS
        ).hour

    def run(self):
        try:
            while True:
                # if there is long interval between 2 events, just keep checking if
                # the file is still open. if it is open for too long and a new event
                # has not occurred, close the open file based on rotation policy.
                if self._writer and self._should_rotate_now(time.time() * CONVERT_TO_MICROSECS):
                    self.close()

                event = self._queue.get()

                if event is self._close_sentinel:
                    return
                elif event is self._flush_sentinel:
                    self.flush()
                    self._flush_complete.set()
                else:

                    if (
                        not self._healthy
                        or not self._profiler_config_parser.profiling_enabled
                        or event is self._sentinel_event
                    ):
                        #self._queue.task_done()
                        break

                    # write event
                    _ = self.write_event(event)
                    now = time.time()
                    if now > self._next_event_flush_time:
                        self.flush()
                        self._next_event_flush_time = now + self._flush_secs
                    #self._queue.task_done()
        except Exception as e:
            #logging.error("EventFileWriter writer thread error: %s", e)
            raise
        finally:
            self._flush_complete.set()
            self._queue.close()


    def open(self, path, cur_event_end_time):
        """
        Open the trace event file either from init or when closing and opening a file based on rotation policy
        """
        try:
            ensure_dir(path)
            self._writer = open(path, "a+")
        except (OSError, IOError) as err:
            logger.debug(f"Sagemaker-Debugger: failed to open {path}: {str(err)}")
            self.continuous_fail_count += 1
            return False
        self.tensor_table = collections.defaultdict(int)
        self.is_first = True
        self._writer.write("[\n")
        self._healthy = True
        self.cur_hour = datetime.utcfromtimestamp(cur_event_end_time / CONVERT_TO_MICROSECS).hour
        return True

    def _get_rotation_info(self, now_in_us):
        file_size = self.file_size()
        now = now_in_us / CONVERT_TO_MICROSECS  # convert to seconds

        # find the difference between the now and last file closed time (in seconds)
        diff_in_seconds = int(round(now - (self.last_file_close_time_in_us / CONVERT_TO_MICROSECS)))

        now_datehour = datetime.utcfromtimestamp(now)

        # check if the flush is going to happen in the next hour, if so,
        # close the file, create a new directory for the next hour and write to file there
        diff_in_hours = abs(now_datehour.hour - self.cur_hour)

        return file_size, diff_in_seconds, diff_in_hours

    def _should_rotate_now(self, now_in_us):
        file_size, diff_in_seconds, diff_in_hours = self._get_rotation_info(now_in_us)
        rotation_policy = self._profiler_config_parser.config.trace_file.rotation_policy

        if diff_in_seconds > rotation_policy.file_close_interval:
            return True

        if file_size > rotation_policy.file_max_size:
            if now_in_us != self.last_event_end_time_in_us:
                self.last_event_end_time_in_us = now_in_us
            return True

        return False

    def write_event(self, record):
        """Appends trace event to the file."""
        # Check if event is of type TimelineRecord.
        if not isinstance(record, TimelineRecord):
            raise TypeError("expected a TimelineRecord, " " but got %s" % type(record))
        self._num_outstanding_events += 1

        """
        Rotation policy:
        Close file if file size exceeds $ENV_MAX_FILE_SIZE or folder was created more than
        $ENV_CLOSE_FILE_INTERVAL time duration.
        """
        end_time_for_event_in_us = record.event_end_ts_micros

        # check if any of the rotation policies have been satisfied. close the existing
        # trace file and open a new one
        # policy 1: if file size exceeds specified max_size
        # policy 2: if same file has been written to for close_interval time
        # policy 3: if a write is being made in the next hour, create a new directory
        if self._writer and self._should_rotate_now(end_time_for_event_in_us):
            self.close()

        #  if file has not been created yet, create now
        if not self._writer:
            file_opened = self.open(path=self.name(), cur_event_end_time=end_time_for_event_in_us)
            if not file_opened:
                file_open_fail_threshold = (
                    self._profiler_config_parser.config.trace_file.file_open_fail_threshold
                )
                if self.continuous_fail_count >= file_open_fail_threshold:
                    logger.warning(
                        "Encountered {} number of continuous failures while trying to open the file. "
                        "Marking the writer unhealthy. All future events will be dropped.".format(
                            str(file_open_fail_threshold)
                        )
                    )
                    self._healthy = False
                return

            # First writing a metadata event
        if self.is_first:
            args = {"start_time_since_epoch_in_micros": record.base_start_time}
            json_dict = {"name": "process_name", "ph": "M", "pid": 0, "args": args}
            self._writer.write(json.dumps(json_dict) + ",\n")

            args = {"sort_index": 0}
            json_dict = {"name": "process_sort_index", "ph": "M", "pid": 0, "args": args}
            self._writer.write(json.dumps(json_dict) + ",\n")
            self.is_first = False

        if self.tensor_table[record.training_phase] == 0:
            # Get the tensor_idx from master table if not create one and append it to master table.
            if record.training_phase in self.training_phase_to_pid:
                tensor_idx = self.training_phase_to_pid[record.training_phase]
            else:
                tensor_idx = len(self.training_phase_to_pid)
                self.training_phase_to_pid[record.training_phase] = tensor_idx

            self.tensor_table[record.training_phase] = tensor_idx

            # Instant events don't have a training phase
            if record.phase != "i":
                args = {"name": record.training_phase}
                json_dict = {"name": "process_name", "ph": "M", "pid": tensor_idx, "args": args}
                self._writer.write(json.dumps(json_dict) + ",\n")

                args = {"sort_index": tensor_idx}
                json_dict = {
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": tensor_idx,
                    "args": args,
                }
                self._writer.write(json.dumps(json_dict) + ",\n")

        record.pid = self.tensor_table[record.training_phase]

        # write the trace event record
        position_and_length_of_record = self._writer.write(record.to_json() + ",\n")
        self.flush()
        if record.event_end_ts_micros > self.last_event_end_time_in_us:
            self.last_event_end_time_in_us = record.event_end_ts_micros
        return position_and_length_of_record

    def flush(self):
        """Flushes the trace event file to disk."""
        if self._num_outstanding_events == 0:
            return
        if self._writer is not None:
            self._writer.flush()
            if self.verbose and logger is not None:
                logger.debug(
                    "wrote %d %s to disk",
                    self._num_outstanding_events,
                    "event" if self._num_outstanding_events == 1 else "events",
                )
            self._num_outstanding_events = 0

    def close(self):
        """Flushes the pending events and closes the writer after it is done."""
        if self._writer is not None:
            # seeking the last ',' and replacing with ']' to mark EOF
            file_seek_pos = self._writer.tell()
            self._writer.seek(file_seek_pos - 2)
            self._writer.truncate()

            if file_seek_pos > 2:
                self._writer.write("\n]")

            self.flush()
            self._writer.close()

            if self._profiler_config_parser.profiling_enabled:
                # ensure that there's a directory for the new file name
                new_file_name = TraceFileLocation().get_file_location(
                    base_dir=self._profiler_config_parser.config.local_path,
                    timestamp=self.last_event_end_time_in_us,
                    suffix=self.suffix,
                )
                ensure_dir(new_file_name)
                os.rename(self.name(), new_file_name)

            self._writer = None
            self.last_file_close_time_in_us = time.time() * CONVERT_TO_MICROSECS

    def name(self):
        return (
            self._profiler_config_parser.config.local_path
            + "/framework/"
            + self.node_id
            + "_"
            + self.suffix
            + SMDEBUG_TEMP_PATH_SUFFIX
        )

    def file_size(self):
        return os.path.getsize(self.name())  # in bytes

class CloseableQueue(object):
  """Stripped-down fork of the standard library Queue that is closeable."""

  def __init__(self, maxsize=0):
    """Create a queue object with a given maximum size.
    Args:
      maxsize: int size of queue. If <= 0, the queue size is infinite.
    """
    self._maxsize = maxsize
    self._queue = collections.deque()
    self._closed = False
    # Mutex must be held whenever queue is mutating; shared by conditions.
    self._mutex = threading.Lock()
    # Notify not_empty whenever an item is added to the queue; a
    # thread waiting to get is notified then.
    self._not_empty = threading.Condition(self._mutex)
    # Notify not_full whenever an item is removed from the queue;
    # a thread waiting to put is notified then.
    self._not_full = threading.Condition(self._mutex)

  def get(self):
    """Remove and return an item from the queue.
    If the queue is empty, blocks until an item is available.
    Returns:
      an item from the queue
    """
    with self._not_empty:
      while not self._queue:
        self._not_empty.wait()
      item = self._queue.popleft()
      self._not_full.notify()
      return item

  def put(self, item):
    """Put an item into the queue.
    If the queue is closed, fails immediately.
    If the queue is full, blocks until space is available or until the queue
    is closed by a call to close(), at which point this call fails.
    Args:
      item: an item to add to the queue
    Raises:
      QueueClosedError: if insertion failed because the queue is closed
    """
    with self._not_full:
      if self._closed:
        raise QueueClosedError()
      if self._maxsize > 0:
        while len(self._queue) == self._maxsize:
          self._not_full.wait()
          if self._closed:
            raise QueueClosedError()
      self._queue.append(item)
      self._not_empty.notify()

  def close(self):
    """Closes the queue, causing any pending or future `put()` calls to fail."""
    with self._not_full:
      self._closed = True
      self._not_full.notify_all()


class QueueClosedError(Exception):
  """Raised when CloseableQueue.put() fails because the queue is closed."""
