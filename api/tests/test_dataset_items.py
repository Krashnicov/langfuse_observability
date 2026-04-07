"""
STORY-014 -- Dataset Items CRUD tests
AC coverage: AC-14.1 through AC-14.5

Pattern: _make_api (direct _client injection, sets _timeout for _sdk_call).
All four methods (create_dataset_item, get_dataset_item, list_dataset_items,
delete_dataset_item) use SDK dispatch via _sdk_call.
Tests mock _sdk_call directly via patch.object for clean isolation.
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
# AC-14.1: create_dataset_item
# ---------------------------------------------------------------------------

class TestCreateDatasetItem(unittest.TestCase):
    """AC-14.1, AC-14.5: create_dataset_item delegates to dataset_items.create via _sdk_call."""

    def test_create_dataset_item_delegates_to_sdk(self):
        """AC-14.1: create_dataset_item calls _sdk_call with dataset_items.create and dataset_name kwarg."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.create_dataset_item('my-dataset')
            # AC-14.1: _sdk_call invoked with correct SDK method and dataset_name as kwarg
            mock_sdk_call.assert_called_once_with(
                api._client.api.dataset_items.create,
                dataset_name='my-dataset'
            )
            self.assertIs(result, mock_sdk_call.return_value)

    def test_create_dataset_item_raises_value_error_on_empty_dataset_name(self):
        """AC-14.1: create_dataset_item('') and create_dataset_item(None) raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-14.1: empty string
            api.create_dataset_item('')
        with self.assertRaises(ValueError):  # AC-14.1: None
            api.create_dataset_item(None)

    def test_create_dataset_item_passes_optional_kwargs(self):
        """AC-14.1: optional kwargs (input, expected_output) forwarded to _sdk_call."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            api.create_dataset_item(
                'ds',
                input={'q': 'What?'},
                expected_output={'a': '42'}
            )
            call_kwargs = mock_sdk_call.call_args.kwargs
            # AC-14.1: all kwargs forwarded
            self.assertEqual(call_kwargs.get('dataset_name'), 'ds')
            self.assertEqual(call_kwargs.get('input'), {'q': 'What?'})
            self.assertEqual(call_kwargs.get('expected_output'), {'a': '42'})

    def test_create_dataset_item_dataset_name_is_kwarg_not_positional(self):
        """AC-14.1: dataset_name MUST appear in call_args.kwargs (not args)."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            api.create_dataset_item('my-dataset')
            # AC-14.1: dataset_name must be keyword, not positional
            call_kwargs = mock_sdk_call.call_args.kwargs
            self.assertIn('dataset_name', call_kwargs)
            self.assertEqual(call_kwargs['dataset_name'], 'my-dataset')
            # dataset_name must NOT appear as extra positional arg
            call_args = mock_sdk_call.call_args.args
            self.assertNotIn('my-dataset', call_args)


# ---------------------------------------------------------------------------
# AC-14.2: get_dataset_item
# ---------------------------------------------------------------------------

class TestGetDatasetItem(unittest.TestCase):
    """AC-14.2, AC-14.5: get_dataset_item delegates to dataset_items.get via _sdk_call."""

    def test_get_dataset_item_delegates_to_sdk(self):
        """AC-14.2: get_dataset_item calls _sdk_call with dataset_items.get and item_id positionally."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.get_dataset_item('item-abc')
            # AC-14.2: item_id passed positionally after the SDK method
            mock_sdk_call.assert_called_once_with(
                api._client.api.dataset_items.get, 'item-abc'
            )
            self.assertIs(result, mock_sdk_call.return_value)

    def test_get_dataset_item_raises_value_error_on_empty_id(self):
        """AC-14.2: get_dataset_item('') and get_dataset_item(None) raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-14.2: empty string
            api.get_dataset_item('')
        with self.assertRaises(ValueError):  # AC-14.2: None
            api.get_dataset_item(None)


# ---------------------------------------------------------------------------
# AC-14.3: list_dataset_items
# ---------------------------------------------------------------------------

class TestListDatasetItems(unittest.TestCase):
    """AC-14.3, AC-14.5: list_dataset_items delegates to dataset_items.list via _sdk_call."""

    def test_list_dataset_items_delegates_to_sdk(self):
        """AC-14.3: list_dataset_items calls _sdk_call with dataset_items.list."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.list_dataset_items()
            # AC-14.3: _sdk_call invoked with correct SDK method
            mock_sdk_call.assert_called_once_with(api._client.api.dataset_items.list)
            self.assertIs(result, mock_sdk_call.return_value)

    def test_list_dataset_items_passes_filter_kwargs(self):
        """AC-14.3: filter kwargs (dataset_name, page, limit) forwarded unchanged to _sdk_call."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            api.list_dataset_items(dataset_name='golden', page=1, limit=50)
            # AC-14.3: kwargs appear in the actual call
            mock_sdk_call.assert_called_once_with(
                api._client.api.dataset_items.list,
                dataset_name='golden',
                page=1,
                limit=50
            )

    def test_list_dataset_items_no_required_args(self):
        """AC-14.3: list_dataset_items() callable with no arguments — no ValueError raised."""
        api = _make_api()
        with patch.object(api, '_sdk_call'):
            # AC-14.3: no required args — should not raise
            try:
                api.list_dataset_items()
            except ValueError:
                self.fail('list_dataset_items() raised ValueError with no args — violates AC-14.3')


# ---------------------------------------------------------------------------
# AC-14.4: delete_dataset_item
# ---------------------------------------------------------------------------

class TestDeleteDatasetItem(unittest.TestCase):
    """AC-14.4, AC-14.5: delete_dataset_item delegates to dataset_items.delete via _sdk_call."""

    def test_delete_dataset_item_delegates_to_sdk(self):
        """AC-14.4: delete_dataset_item calls _sdk_call with dataset_items.delete and item_id positionally."""
        api = _make_api()
        with patch.object(api, '_sdk_call') as mock_sdk_call:
            result = api.delete_dataset_item('item-xyz')
            # AC-14.4: item_id passed positionally after the SDK method
            mock_sdk_call.assert_called_once_with(
                api._client.api.dataset_items.delete, 'item-xyz'
            )
            self.assertIs(result, mock_sdk_call.return_value)

    def test_delete_dataset_item_raises_value_error_on_empty_id(self):
        """AC-14.4: delete_dataset_item('') and delete_dataset_item(None) raise ValueError."""
        api = _make_api()
        with self.assertRaises(ValueError):  # AC-14.4: empty string
            api.delete_dataset_item('')
        with self.assertRaises(ValueError):  # AC-14.4: None
            api.delete_dataset_item(None)


# ---------------------------------------------------------------------------
# AC-14.5: Error propagation
# ---------------------------------------------------------------------------

class TestDatasetItemsErrorPropagation(unittest.TestCase):
    """AC-14.5: All methods propagate SDK errors via _sdk_call translation."""

    def test_dataset_items_propagate_api_error(self):
        """AC-14.5: LangfuseAPIError raised by _sdk_call propagates from list_dataset_items."""
        api = _make_api()
        with patch.object(
            api, '_sdk_call', side_effect=LangfuseAPIError(500, 'Server Error')
        ):
            with self.assertRaises(LangfuseAPIError) as ctx:  # AC-14.5
                api.list_dataset_items()
            self.assertEqual(ctx.exception.status_code, 500)  # AC-14.5: status_code preserved


if __name__ == '__main__':
    unittest.main()
