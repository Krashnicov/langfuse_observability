"""
STORY-015 -- Dataset Runs & Run Items tests
AC coverage: AC-15.1 through AC-15.6

Pattern: _make_api (direct _client injection, sets _timeout for _sdk_call).
All five methods use SDK dispatch via _sdk_call.
Tests mock _sdk_call directly via patch.object for clean isolation.

SDK sub-clients:
  - self._client.api.datasets       → get_run, delete_run, get_runs
  - self._client.api.dataset_run_items → create, list
"""
import unittest
from unittest.mock import MagicMock, patch

from api.langfuse_observability_api import LangfuseObservabilityAPI
from api.langfuse_client import LangfuseAPIError, LangfuseAuthError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_api(mock_client=None):
    """Construct LangfuseObservabilityAPI bypassing __init__ (no SDK singleton needed)."""
    api = LangfuseObservabilityAPI.__new__(LangfuseObservabilityAPI)
    api._client = mock_client if mock_client is not None else MagicMock()
    api._timeout = LangfuseObservabilityAPI.DEFAULT_TIMEOUT  # required by _sdk_call
    return api


# ---------------------------------------------------------------------------
# AC-15.1: get_dataset_run
# ---------------------------------------------------------------------------

class TestGetDatasetRun(unittest.TestCase):
    """AC-15.1, AC-15.6: get_dataset_run delegates to datasets.get_run via _sdk_call."""

    def test_get_dataset_run_delegates_to_sdk(self):
        """AC-15.1: get_dataset_run calls _sdk_call with datasets.get_run and both positional args."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.get_dataset_run('my-ds', 'run-1')
            # AC-15.1: _sdk_call invoked with correct SDK method and both positional args
            mock_sdk_call.assert_called_once_with(
                api._client.api.datasets.get_run,
                'my-ds',
                'run-1'
            )
            self.assertIs(result, mock_sdk_call.return_value)

    def test_get_dataset_run_raises_value_error_on_empty_dataset_name(self):
        """AC-15.1: get_dataset_run('', 'run-1') and get_dataset_run(None, 'run-1') raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-15.1: empty string
            api.get_dataset_run('', 'run-1')
        with self.assertRaises(ValueError):  # AC-15.1: None
            api.get_dataset_run(None, 'run-1')

    def test_get_dataset_run_raises_value_error_on_empty_run_name(self):
        """AC-15.1: get_dataset_run('ds', '') and get_dataset_run('ds', None) raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-15.1: empty string
            api.get_dataset_run('ds', '')
        with self.assertRaises(ValueError):  # AC-15.1: None
            api.get_dataset_run('ds', None)


# ---------------------------------------------------------------------------
# AC-15.2: delete_dataset_run
# ---------------------------------------------------------------------------

class TestDeleteDatasetRun(unittest.TestCase):
    """AC-15.2, AC-15.6: delete_dataset_run delegates to datasets.delete_run via _sdk_call."""

    def test_delete_dataset_run_delegates_to_sdk(self):
        """AC-15.2: delete_dataset_run calls _sdk_call with datasets.delete_run and both positional args."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.delete_dataset_run('my-ds', 'old-run')
            # AC-15.2: _sdk_call invoked with correct SDK method and both positional args
            mock_sdk_call.assert_called_once_with(
                api._client.api.datasets.delete_run,
                'my-ds',
                'old-run'
            )
            self.assertIs(result, mock_sdk_call.return_value)

    def test_delete_dataset_run_raises_value_error_on_falsy_args(self):
        """AC-15.2: ValueError raised when either dataset_name or run_name is falsy."""
        api = _make_api()
        # empty dataset_name
        with self.assertRaises(ValueError):  # AC-15.2: empty dataset_name
            api.delete_dataset_run('', 'old-run')
        with self.assertRaises(ValueError):  # AC-15.2: None dataset_name
            api.delete_dataset_run(None, 'old-run')
        # empty run_name
        with self.assertRaises(ValueError):  # AC-15.2: empty run_name
            api.delete_dataset_run('my-ds', '')
        with self.assertRaises(ValueError):  # AC-15.2: None run_name
            api.delete_dataset_run('my-ds', None)


# ---------------------------------------------------------------------------
# AC-15.3: list_dataset_runs
# ---------------------------------------------------------------------------

class TestListDatasetRuns(unittest.TestCase):
    """AC-15.3, AC-15.6: list_dataset_runs delegates to datasets.get_runs via _sdk_call."""

    def test_list_dataset_runs_delegates_to_sdk(self):
        """AC-15.3: list_dataset_runs calls _sdk_call with datasets.get_runs and dataset_name positional."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.list_dataset_runs('my-ds')
            # AC-15.3: _sdk_call invoked with correct SDK method and dataset_name positional
            mock_sdk_call.assert_called_once_with(
                api._client.api.datasets.get_runs,
                'my-ds'
            )
            self.assertIs(result, mock_sdk_call.return_value)

    def test_list_dataset_runs_passes_page_limit_kwargs(self):
        """AC-15.3: optional page and limit kwargs forwarded to _sdk_call."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            api.list_dataset_runs('ds', page=2, limit=10)
            # AC-15.3: page and limit forwarded as kwargs
            mock_sdk_call.assert_called_once_with(
                api._client.api.datasets.get_runs,
                'ds',
                page=2,
                limit=10
            )

    def test_list_dataset_runs_raises_value_error_on_empty_dataset_name(self):
        """AC-15.3: list_dataset_runs('') and list_dataset_runs(None) raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-15.3: empty string
            api.list_dataset_runs('')
        with self.assertRaises(ValueError):  # AC-15.3: None
            api.list_dataset_runs(None)


# ---------------------------------------------------------------------------
# AC-15.4: create_dataset_run_item
# ---------------------------------------------------------------------------

class TestCreateDatasetRunItem(unittest.TestCase):
    """AC-15.4, AC-15.6: create_dataset_run_item delegates to dataset_run_items.create via _sdk_call."""

    def test_create_dataset_run_item_delegates_to_sdk(self):
        """AC-15.4: create_dataset_run_item calls _sdk_call with run_name and dataset_item_id as kwargs."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.create_dataset_run_item('run-1', 'item-abc')
            # AC-15.4: _sdk_call invoked with correct SDK method, both required as kwargs
            mock_sdk_call.assert_called_once_with(
                api._client.api.dataset_run_items.create,
                run_name='run-1',
                dataset_item_id='item-abc'
            )
            self.assertIs(result, mock_sdk_call.return_value)

    def test_create_dataset_run_item_raises_value_error_on_empty_run_name(self):
        """AC-15.4: create_dataset_run_item('', 'item-abc') and (None, 'item-abc') raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-15.4: empty run_name
            api.create_dataset_run_item('', 'item-abc')
        with self.assertRaises(ValueError):  # AC-15.4: None run_name
            api.create_dataset_run_item(None, 'item-abc')

    def test_create_dataset_run_item_raises_value_error_on_empty_item_id(self):
        """AC-15.4: create_dataset_run_item('run-1', '') and ('run-1', None) raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-15.4: empty dataset_item_id
            api.create_dataset_run_item('run-1', '')
        with self.assertRaises(ValueError):  # AC-15.4: None dataset_item_id
            api.create_dataset_run_item('run-1', None)


# ---------------------------------------------------------------------------
# AC-15.5: list_dataset_run_items
# ---------------------------------------------------------------------------

class TestListDatasetRunItems(unittest.TestCase):
    """AC-15.5, AC-15.6: list_dataset_run_items delegates to dataset_run_items.list via _sdk_call."""

    def test_list_dataset_run_items_delegates_to_sdk(self):
        """AC-15.5: list_dataset_run_items calls _sdk_call with dataset_id and run_name as kwargs."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.list_dataset_run_items('ds-id-123', 'run-1')
            # AC-15.5: _sdk_call invoked with dataset_run_items.list, both required as kwargs
            # CRITICAL: dataset_id NOT dataset_name
            mock_sdk_call.assert_called_once_with(
                api._client.api.dataset_run_items.list,
                dataset_id='ds-id-123',
                run_name='run-1'
            )
            self.assertIs(result, mock_sdk_call.return_value)


if __name__ == '__main__':
    unittest.main()
