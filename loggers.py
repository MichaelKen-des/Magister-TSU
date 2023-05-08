import streamlit as st
from tensorflow import keras
from keras.utils.generic_utils import Progbar
import time
import numpy as np
import sys
import os



class CustomProgbar(Progbar):
  """Displays a progress bar.

   Args:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that should *not*
          be averaged over time. Metrics in this list will be displayed as-is.
          All others will be averaged by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
        unit_name: Display name for step counts (usually "step" or "sample").
  """
  def __init__(
        self,
        target,
        container,
        width=30,
        verbose=1,
        interval=0.05,
        stateful_metrics=None,
        unit_name="step",
    ):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = (
            (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            or "ipykernel" in sys.modules
            or "posix" in sys.modules
            or "PYCHARM_HOSTED" in os.environ
        )
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0
        self._time_at_epoch_start = self._start
        self._time_at_epoch_end = None
        self._time_after_first_step = None
        self.container = container

  def update(self, current, values=None, finalize=None):
    """Updates the progress bar.

    Args:
        current: Index of current step.
        values: List of tuples: `(name, value_for_last_step)`. If `name` is in
          `stateful_metrics`, `value_for_last_step` will be displayed as-is.
          Else, an average of the metric over time will be displayed.
        finalize: Whether this is the last update for the progress bar. If
          `None`, defaults to `current >= self.target`.
    """
    if finalize is None:
      if self.target is None:
        finalize = False
      else:
        finalize = current >= self.target

    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        # In the case that progress bar doesn't have a target value in the first
        # epoch, both on_batch_end and on_epoch_end will be called, which will
        # cause 'current' and 'self._seen_so_far' to have the same value. Force
        # the minimal value to 1 here, otherwise stateful_metric will be 0s.
        value_base = max(current - self._seen_so_far, 1)
        if k not in self._values:
          self._values[k] = [v * value_base, value_base]
        else:
          self._values[k][0] += v * value_base
          self._values[k][1] += value_base
      else:
        # Stateful metrics output a numeric value. This representation
        # means "take an average from a single value" but keeps the
        # numeric formatting.
        self._values[k] = [v, 1]
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start)
    if self.verbose == 1:
      if now - self._last_update < self.interval and not finalize:
        return

      prev_total_width = self._total_width

      time_per_unit = self._estimate_step_duration(current, now)

      if self.target is None or finalize:
        if time_per_unit >= 1 or time_per_unit == 0:
          info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
        else:
          info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)
      else:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600,
                                         (eta % 3600) // 60, eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta

        info = ' - ETA: %s' % eta_format

      for k in self._values_order:
        info += ' - %s:' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))

      if finalize:
        info += '\n'

      self.container.write(info)

    elif self.verbose == 2:
      if finalize:
        numdigits = int(np.log10(self.target)) + 1
        count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
        info = count + info
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        info += '\n'

        self.container.write(info)

    self._last_update = now


class CustomProgbarLogger(keras.callbacks.ProgbarLogger):
    """Callback that prints metrics to UI.
    Args:
        count_mode: One of `"steps"` or `"samples"`.
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).
            If not provided, defaults to the `Model`'s metrics.
    Raises:
        ValueError: In case of invalid `count_mode`.
    """
    def __init__(self, container_epoch, container, count_mode: str = "samples", stateful_metrics=None):
        super().__init__()
        self._supports_tf_logs = True
        if count_mode == "samples":
            self.use_steps = False
        elif count_mode == "steps":
            self.use_steps = True
        else:
            raise ValueError(
                f"Unknown `count_mode`: {count_mode}. "
                'Expected values are ["samples", "steps"]'
            )
        # Defaults to all Model's metrics except for loss.
        self.stateful_metrics = (
            set(stateful_metrics) if stateful_metrics else set()
        )

        self.seen = 0
        self.progbar = None
        self.target = None
        self.verbose = 1
        self.epochs = 1

        self._train_step, self._test_step, self._predict_step = None, None, None
        self._call_batch_hooks = True

        self._called_in_fit = False
        self.container_epoch = container_epoch
        self.container = container
    
    def on_epoch_begin(self, epoch, logs=None):
        self._reset_progbar()
        self._maybe_init_progbar()
        if self.verbose and self.epochs > 1:
            self.container_epoch.write(f"Epoch {epoch + 1}/{self.epochs}")
            # Можно передать сюда прогресс-бар

    def _maybe_init_progbar(self):
        """Instantiate a `Progbar` if not yet, and update the stateful
        metrics."""
        # TODO(rchao): Legacy TF1 code path may use list for
        # `self.stateful_metrics`. Remove "cast to set" when TF1 support is
        # dropped.
        self.stateful_metrics = set(self.stateful_metrics)

        if self.model:
            # Update the existing stateful metrics as `self.model.metrics` may
            # contain updated metrics after `MetricsContainer` is built in the
            # first train step.
            self.stateful_metrics = self.stateful_metrics.union(
                set(m.name for m in self.model.metrics)
            )

        if self.progbar is None:
            self.progbar = CustomProgbar(
                target=self.target,
                verbose=self.verbose,
                stateful_metrics=self.stateful_metrics,
                unit_name="step" if self.use_steps else "sample",
                container=self.container
            )

        self.progbar._update_stateful_metrics(self.stateful_metrics)