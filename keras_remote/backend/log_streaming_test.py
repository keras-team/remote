"""Tests for keras_remote.backend.log_streaming — live pod log streaming."""

import io
import threading
from collections import deque
from unittest import mock
from unittest.mock import MagicMock

import urllib3
from absl.testing import absltest
from kubernetes.client.rest import ApiException

from keras_remote.backend.log_streaming import (
  _make_log_panel,
  _render_live_panel,
  _render_plain,
  _start_log_streaming,
  _stream_pod_logs,
)


class TestStreamPodLogs(absltest.TestCase):
  """Tests for the top-level _stream_pod_logs orchestrator."""

  def _make_mock_resp(self, chunks):
    mock_resp = MagicMock()
    mock_resp.stream.return_value = chunks
    return mock_resp

  def test_calls_log_api_correctly(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.return_value = self._make_mock_resp([])

    with mock.patch(
      "keras_remote.backend.log_streaming.Console"
    ) as mock_console_cls:
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default")

    mock_core.read_namespaced_pod_log.assert_called_once_with(
      name="pod-1",
      namespace="default",
      follow=True,
      _preload_content=False,
    )

  def test_uses_live_panel_for_terminal(self):
    mock_core = MagicMock()
    mock_resp = self._make_mock_resp([b"hello\n"])
    mock_core.read_namespaced_pod_log.return_value = mock_resp

    with (
      mock.patch(
        "keras_remote.backend.log_streaming.Console"
      ) as mock_console_cls,
      mock.patch(
        "keras_remote.backend.log_streaming._render_live_panel"
      ) as mock_live,
      mock.patch(
        "keras_remote.backend.log_streaming._render_plain"
      ) as mock_plain,
    ):
      mock_console_cls.return_value.is_terminal = True
      _stream_pod_logs(mock_core, "pod-1", "default")

    mock_live.assert_called_once()
    mock_plain.assert_not_called()

  def test_uses_plain_for_non_terminal(self):
    mock_core = MagicMock()
    mock_resp = self._make_mock_resp([b"hello\n"])
    mock_core.read_namespaced_pod_log.return_value = mock_resp

    with (
      mock.patch(
        "keras_remote.backend.log_streaming.Console"
      ) as mock_console_cls,
      mock.patch(
        "keras_remote.backend.log_streaming._render_live_panel"
      ) as mock_live,
      mock.patch(
        "keras_remote.backend.log_streaming._render_plain"
      ) as mock_plain,
    ):
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default")

    mock_plain.assert_called_once()
    mock_live.assert_not_called()

  def test_releases_conn_on_api_exception(self):
    mock_core = MagicMock()
    mock_resp = MagicMock()
    mock_resp.stream.side_effect = ApiException(status=404, reason="Not Found")
    mock_core.read_namespaced_pod_log.return_value = mock_resp

    with mock.patch(
      "keras_remote.backend.log_streaming.Console"
    ) as mock_console_cls:
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default")

    mock_resp.release_conn.assert_called_once()

  def test_suppresses_protocol_error(self):
    mock_core = MagicMock()
    mock_resp = MagicMock()
    mock_resp.stream.side_effect = urllib3.exceptions.ProtocolError(
      "Connection broken"
    )
    mock_core.read_namespaced_pod_log.return_value = mock_resp

    with mock.patch(
      "keras_remote.backend.log_streaming.Console"
    ) as mock_console_cls:
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default")

    mock_resp.release_conn.assert_called_once()

  def test_handles_error_before_response(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.side_effect = ApiException(
      status=404, reason="Not Found"
    )

    # Should not raise even when resp is None
    _stream_pod_logs(mock_core, "pod-1", "default")

  def test_logs_warning_on_unexpected_error(self):
    mock_core = MagicMock()
    mock_resp = MagicMock()
    mock_resp.stream.side_effect = ValueError("something unexpected")
    mock_core.read_namespaced_pod_log.return_value = mock_resp

    with (
      mock.patch(
        "keras_remote.backend.log_streaming.Console"
      ) as mock_console_cls,
      mock.patch("keras_remote.backend.log_streaming.logging") as mock_log,
    ):
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default")

    mock_log.warning.assert_called_once()
    self.assertIn("pod-1", mock_log.warning.call_args[0][1])
    mock_resp.release_conn.assert_called_once()


class TestRenderPlain(absltest.TestCase):
  """Tests for the non-terminal plain rendering path."""

  def test_streams_chunks_to_stdout(self):
    mock_resp = MagicMock()
    mock_resp.stream.return_value = [b"line 1\n", b"line 2\n"]
    console = MagicMock()

    with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
      _render_plain(mock_resp, "pod-1", console)

    self.assertIn("line 1", mock_stdout.getvalue())
    self.assertIn("line 2", mock_stdout.getvalue())

  def test_prints_rule_delimiters(self):
    mock_resp = MagicMock()
    mock_resp.stream.return_value = []
    console = MagicMock()

    _render_plain(mock_resp, "pod-1", console)

    self.assertEqual(console.rule.call_count, 2)
    # Opening rule contains pod name
    self.assertIn("pod-1", console.rule.call_args_list[0][0][0])

  def test_handles_utf8_decode_errors(self):
    mock_resp = MagicMock()
    mock_resp.stream.return_value = [b"valid\n", b"\xff\xfe invalid\n"]
    console = MagicMock()

    with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
      _render_plain(mock_resp, "pod-1", console)

    output = mock_stdout.getvalue()
    self.assertIn("valid", output)
    self.assertIn("invalid", output)


class TestRenderLivePanel(absltest.TestCase):
  """Tests for the terminal Live panel rendering path."""

  def test_updates_panel_with_log_lines(self):
    mock_resp = MagicMock()
    mock_resp.stream.return_value = [b"line 1\n", b"line 2\n"]
    console = MagicMock()

    with mock.patch("keras_remote.backend.log_streaming.Live") as mock_live_cls:
      mock_live = MagicMock()
      mock_live_cls.return_value.__enter__ = MagicMock(return_value=mock_live)
      mock_live_cls.return_value.__exit__ = MagicMock(return_value=False)
      _render_live_panel(mock_resp, "pod-1", console)

    # Panel should have been updated at least once
    self.assertTrue(mock_live.update.called)

  def test_handles_partial_lines(self):
    mock_resp = MagicMock()
    # "hello\nwor" then "ld\n" — "world" is split across chunks
    mock_resp.stream.return_value = [b"hello\nwor", b"ld\n"]
    console = MagicMock()

    with mock.patch("keras_remote.backend.log_streaming.Live") as mock_live_cls:
      mock_live = MagicMock()
      mock_live_cls.return_value.__enter__ = MagicMock(return_value=mock_live)
      mock_live_cls.return_value.__exit__ = MagicMock(return_value=False)
      _render_live_panel(mock_resp, "pod-1", console)

    # Check the final panel contains both complete lines
    last_panel = mock_live.update.call_args_list[-1][0][0]
    panel_content = last_panel.renderable
    self.assertIn("hello", panel_content)
    self.assertIn("world", panel_content)


class TestMakeLogPanel(absltest.TestCase):
  def test_empty_shows_waiting(self):
    panel = _make_log_panel(deque(), "title")
    self.assertEqual(panel.renderable, "Waiting for output...")
    self.assertEqual(panel.title, "title")

  def test_joins_lines(self):
    lines = deque(["line 1", "line 2"])
    panel = _make_log_panel(lines, "title")
    self.assertEqual(panel.renderable, "line 1\nline 2")

  def test_border_style(self):
    panel = _make_log_panel(deque(), "t")
    self.assertEqual(panel.border_style, "blue")


class TestStartLogStreaming(absltest.TestCase):
  def test_starts_daemon_thread(self):
    mock_core = MagicMock()

    with mock.patch(
      "keras_remote.backend.log_streaming._stream_pod_logs"
    ) as mock_stream:
      thread = _start_log_streaming(mock_core, "pod-1", "default")

    self.assertIsInstance(thread, threading.Thread)
    self.assertTrue(thread.daemon)
    thread.join(timeout=2)
    mock_stream.assert_called_once_with(mock_core, "pod-1", "default")

  def test_returns_thread_handle(self):
    mock_core = MagicMock()

    with mock.patch("keras_remote.backend.log_streaming._stream_pod_logs"):
      thread = _start_log_streaming(mock_core, "pod-1", "ns")

    self.assertIsNotNone(thread)
    thread.join(timeout=2)


if __name__ == "__main__":
  absltest.main()
