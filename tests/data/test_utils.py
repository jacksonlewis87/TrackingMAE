import pytest
from unittest.mock import create_autospec, patch

from constants import EVAL_GAME_IDS
from data.mae.data_config import DataConfig
from data.utils import create_data_split, get_data_split, get_eval_game_ids, get_non_eval_game_ids


@pytest.mark.parametrize("stage, is_file_result", [("eval", None), ("not_eval", True), ("not_eval", False)])
@patch("data.utils.get_eval_game_ids")
@patch("data.utils.os.path.isfile")
@patch("data.utils.load_json")
@patch("data.utils.create_data_split")
def test_get_data_split(
    mock_create_data_split, mock_load_json, mock_isfile, mock_get_eval_game_ids, stage, is_file_result
):
    mock_config = create_autospec(DataConfig)
    mock_config.data_split_path = "mock_data_split_path"
    mock_game_ids = ["mock_game_id"]
    mock_isfile.return_value = is_file_result

    result = get_data_split(config=mock_config, game_ids=mock_game_ids, stage=stage)

    if stage == "eval":
        mock_get_eval_game_ids.assert_called_once_with(game_ids=mock_game_ids)
        expected_result = {"train": [], "val": mock_get_eval_game_ids.return_value}
    else:
        mock_isfile.assert_called_once_with(mock_config.data_split_path)
        if is_file_result:
            mock_load_json.assert_called_once_with(path=mock_config.data_split_path)
            expected_result = mock_load_json.return_value
        else:
            mock_create_data_split.assert_called_once_with(config=mock_config, game_ids=mock_game_ids)
            expected_result = mock_create_data_split.return_value

    assert result == expected_result


@patch("data.utils.get_non_eval_game_ids")
@patch("data.utils.shuffle")
@patch("data.utils.write_json")
def test_create_data_split(mock_write_json, mock_shuffle, mock_get_non_eval_game_ids):
    mock_config = create_autospec(DataConfig)
    mock_config.data_split_path = "mock_data_split_path"
    mock_config.train_size = 0.75
    mock_game_ids = ["mock_game_id"]
    mock_get_non_eval_game_ids.return_value = ["id_0", "id_1", "id_2", "id_3"]
    expected_result = {
        "train": ["id_0", "id_1", "id_2"],
        "val": ["id_3"],
    }

    result = create_data_split(config=mock_config, game_ids=mock_game_ids)

    mock_get_non_eval_game_ids.assert_called_once_with(game_ids=mock_game_ids)
    mock_shuffle.assert_called_once()
    mock_write_json.assert_called_once_with(mock_config.data_split_path, expected_result)

    assert result == expected_result


def test_get_non_eval_game_ids():
    game_ids = [EVAL_GAME_IDS[0], "non_eval_game_id"]

    result = get_non_eval_game_ids(game_ids=game_ids)

    assert result == ["non_eval_game_id"]


def test_get_eval_game_ids():
    game_ids = [EVAL_GAME_IDS[0], "non_eval_game_id"]

    result = get_eval_game_ids(game_ids=game_ids)

    assert result == [EVAL_GAME_IDS[0]]
