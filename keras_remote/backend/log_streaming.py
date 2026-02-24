"""Live log streaming from Kubernetes pods.

Provides utilities to stream pod logs to stdout in real-time using a
background daemon thread. Used by both GKE and Pathways backends during
job execution.
"""

import sys
import threading
from collections import deque

import urllib3
from absl import logging
from kubernetes.client.rest import ApiException
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

_MAX_DISPLAY_LINES = 25


def _stream_pod_logs(core_v1, pod_name, namespace):
  """Stream pod logs to stdout. Designed to run in a daemon thread.

  Uses the Kubernetes follow API to tail logs in real-time. The stream
  ends naturally when the container exits.

  In interactive terminals, logs are displayed in a Rich Live panel.
  In non-interactive contexts (piped output, CI), logs are streamed
  as raw lines with Rich Rule delimiters.

  Args:
      core_v1: Kubernetes CoreV1Api client.
      pod_name: Name of the pod to stream logs from.
      namespace: Kubernetes namespace.
  """
  console = Console()
  resp = None
  try:
    resp = core_v1.read_namespaced_pod_log(
      name=pod_name,
      namespace=namespace,
      follow=True,
      _preload_content=False,
    )
    if console.is_terminal:
      _render_live_panel(resp, pod_name, console)
    else:
      _render_plain(resp, pod_name, console)
  except ApiException:
    pass  # Pod deleted or not found
  except urllib3.exceptions.ProtocolError:
    pass  # Connection broken mid-stream (pod terminated)
  except Exception:
    logging.warning(
      "Log streaming from %s failed unexpectedly", pod_name, exc_info=True
    )
  finally:
    if resp is not None:
      resp.release_conn()


def _render_live_panel(resp, pod_name, console):
  """Render streaming logs inside a Rich Live panel."""
  lines = deque(maxlen=_MAX_DISPLAY_LINES)
  title = f"Remote logs \u2022 {pod_name}"
  buffer = ""

  with Live(
    _make_log_panel(lines, title),
    console=console,
    refresh_per_second=4,
  ) as live:
    for chunk in resp.stream(decode_content=True):
      buffer += chunk.decode("utf-8", errors="replace")
      while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        lines.append(line)
      live.update(_make_log_panel(lines, title))

    # Flush remaining partial line
    if buffer.strip():
      lines.append(buffer)
      live.update(_make_log_panel(lines, title))


def _render_plain(resp, pod_name, console):
  """Render streaming logs as raw lines with Rule delimiters."""
  console.rule(f"Remote logs ({pod_name})", style="blue")
  for chunk in resp.stream(decode_content=True):
    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
    sys.stdout.flush()
  console.rule("End remote logs", style="blue")


def _make_log_panel(lines, title):
  """Build a Panel renderable from accumulated log lines."""
  content = "\n".join(lines) if lines else "Waiting for output..."
  return Panel(content, title=title, border_style="blue")


def _start_log_streaming(core_v1, pod_name, namespace):
  """Start streaming pod logs in a background daemon thread.

  Args:
      core_v1: Kubernetes CoreV1Api client.
      pod_name: Name of the pod to stream logs from.
      namespace: Kubernetes namespace.

  Returns:
      threading.Thread: The daemon thread streaming logs.
  """
  logging.info("Streaming logs from %s...", pod_name)
  thread = threading.Thread(
    target=_stream_pod_logs,
    args=(core_v1, pod_name, namespace),
    daemon=True,
  )
  thread.start()
  return thread
