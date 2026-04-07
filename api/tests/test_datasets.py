"""
STORY-013 -- Datasets CRUD tests
AC coverage: AC-13.1 through AC-13.5

Pattern: _make_api (direct _client injection, sets _timeout for _sdk_call).
All three methods (list_datasets, get_dataset, create_dataset) use SDK dispatch
via _sdk_call. Tests mock _sdk_call directly via patch.object for clean isolation.
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
# AC-13.1: list_datasets
# ---------------------------------------------------------------------------

class TestListDatasets(unittest.TestCase):
    """AC-13.1, AC-13.4, AC-13.5: list_datasets delegates to datasets.list via _sdk_call."""

    def test_list_datasets_delegates_to_sdk(self):
        """AC-13.1: list_datasets calls _sdk_call with datasets.list as the first arg."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.list_datasets()
            # AC-13.1: _sdk_call invoked with the correct SDK method
            mock_sdk_call.assert_called_once_with(api._client.api.datasets.list)
            self.assertIs(result, mock_sdk_call.return_value)

    def test_list_datasets_passes_page_limit_kwargs(self):
        """AC-13.1: page and limit kwargs forwarded unchanged to _sdk_call."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            api.list_datasets(page=2, limit=20)
            # AC-13.1: kwargs appear in the actual call
            mock_sdk_call.assert_called_once_with(
                api._client.api.datasets.list, page=2, limit=20
            )

    def test_list_datasets_raises_auth_error_on_401(self):
        """AC-13.4: LangfuseAuthError raised by _sdk_call propagates to caller."""
        api = _make_api()
        with patch.object(
            api, '_sdk_call', side_effect=LangfuseAuthError(401, 'Unauthorized')
        ):
            with self.assertRaises(LangfuseAuthError):  # AC-13.4
                api.list_datasets()


# ---------------------------------------------------------------------------
# AC-13.2: get_dataset
# ---------------------------------------------------------------------------

class TestGetDataset(unittest.TestCase):
    """AC-13.2, AC-13.4, AC-13.5: get_dataset delegates to datasets.get via _sdk_call."""

    def test_get_dataset_delegates_to_sdk(self):
        """AC-13.2: get_dataset calls _sdk_call with datasets.get and dataset_name."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.get_dataset('my-dataset')
            # AC-13.2: positional arg must be 'my-dataset'
            mock_sdk_call.assert_called_once_with(
                api._client.api.datasets.get, 'my-dataset'
            )
            self.assertIs(result, mock_sdk_call.return_value)

    def test_get_dataset_raises_value_error_on_empty_name(self):
        """AC-13.2: get_dataset('') and get_dataset(None) raise ValueError before SDK call."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-13.2: empty string
            api.get_dataset('')
        with self.assertRaises(ValueError):  # AC-13.2: None
            api.get_dataset(None)

    def test_get_dataset_raises_api_error_on_404(self):
        """AC-13.4: LangfuseAPIError with status_code=404 raised by _sdk_call propagates."""
        api = _make_api()
        with patch.object(
            api, '_sdk_call', side_effect=LangfuseAPIError(404, 'Not found')
        ):
            with self.assertRaises(LangfuseAPIError) as ctx:  # AC-13.4
                api.get_dataset('missing-ds')
            self.assertEqual(ctx.exception.status_code, 404)  # AC-13.4: status_code preserved


# ---------------------------------------------------------------------------
# AC-13.3: create_dataset
# ---------------------------------------------------------------------------

class TestCreateDataset(unittest.TestCase):
    """AC-13.3, AC-13.4, AC-13.5: create_dataset delegates to datasets.create via _sdk_call."""

    def test_create_dataset_delegates_to_sdk(self):
        """AC-13.3: create_dataset calls _sdk_call with datasets.create and name='golden-set'."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.create_dataset('golden-set')
            # AC-13.3: name forwarded as kwarg
            mock_sdk_call.assert_called_once_with(
                api._client.api.datasets.create, name='golden-set'
            )
            self.assertIs(result, mock_sdk_call.return_value)

    def test_create_dataset_raises_value_error_on_empty_name(self):
        """AC-13.3: create_dataset('') and create_dataset(None) raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-13.3: empty string
            api.create_dataset('')
        with self.assertRaises(ValueError):  # AC-13.3: None
            api.create_dataset(None)

    def test_create_dataset_passes_optional_kwargs(self):
        """AC-13.3: optional kwargs (description, metadata) forwarded to _sdk_call."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            api.create_dataset('ds', description='test desc', metadata={'env': 'ci'})
            call_kwargs = mock_sdk_call.call_args.kwargs
            # AC-13.3: all kwargs forwarded
            self.assertEqual(call_kwargs.get('name'), 'ds')
            self.assertEqual(call_kwargs.get('description'), 'test desc')
            self.assertEqual(call_kwargs.get('metadata'), {'env': 'ci'})

    def test_create_dataset_raises_auth_error_on_401(self):
        """AC-13.4: LangfuseAuthError raised by _sdk_call propagates to caller."""
        api = _make_api()
        with patch.object(
            api, '_sdk_call', side_effect=LangfuseAuthError(401, 'Unauthorized')
        ):
            with self.assertRaises(LangfuseAuthError):  # AC-13.4
                api.create_dataset('ds')


if __name__ == '__main__':
    unittest.main()
