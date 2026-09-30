# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import pytest

from rapidocr import LangRec, ModelType, OCRVersion
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapidocr.utils.model_resolver import (
    list_supported_langs,
    model_lang_prefix,
    resolve_model_key,
)
from rapidocr.utils.typings import EngineType, TaskType


@pytest.mark.parametrize(
    "task_type,expected_key",
    [
        (TaskType.DET, "multi_PP-OCRv6_det_tiny"),
        (TaskType.REC, "multi_PP-OCRv6_rec_tiny"),
    ],
)
def test_ppocrv6_tiny_supports_non_japan_lang(task_type, expected_key):
    model_key = resolve_model_key(task_type, OCRVersion.PPOCRV6, "fr", ModelType.TINY)

    assert model_key == expected_key


@pytest.mark.parametrize("model_type", [ModelType.SMALL, ModelType.MEDIUM])
def test_ppocrv6_japan_supported_by_non_tiny_models(model_type):
    model_key = resolve_model_key(TaskType.REC, OCRVersion.PPOCRV6, "japan", model_type)

    assert model_key == f"multi_PP-OCRv6_rec_{model_type.value}"


@pytest.mark.parametrize("task_type", [TaskType.DET, TaskType.REC])
def test_ppocrv6_tiny_does_not_support_japan(task_type):
    with pytest.raises(ValueError, match="japan.*PP-OCRv6 tiny"):
        resolve_model_key(task_type, OCRVersion.PPOCRV6, "japan", ModelType.TINY)


def test_ppocrv6_tiny_rejects_japan_alias():
    with pytest.raises(ValueError, match="japan.*PP-OCRv6 tiny"):
        resolve_model_key(TaskType.REC, OCRVersion.PPOCRV6, "ja", ModelType.TINY)


def test_list_supported_langs_can_filter_by_model_type():
    tiny_langs = list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6, ModelType.TINY)
    small_langs = list_supported_langs(
        TaskType.REC, OCRVersion.PPOCRV6, ModelType.SMALL
    )
    all_langs = list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6)

    assert "japan" not in tiny_langs
    assert "japan" in small_langs
    assert "japan" in all_langs


def test_resolve_model_key_returns_none_for_unrouted_versions():
    model_key = resolve_model_key(
        TaskType.REC, OCRVersion.PPOCRV5, "ch", ModelType.MOBILE
    )

    assert model_key is None


@pytest.mark.parametrize(
    "lang_type", ["id", "indonesian", "bahasa_indonesia", LangRec.ID]
)
@pytest.mark.parametrize(
    "task_type,expected_key",
    [
        (TaskType.DET, "multi_PP-OCRv6_det_small"),
        (TaskType.REC, "multi_PP-OCRv6_rec_small"),
    ],
)
def test_indonesian_is_supported_by_default_ppocrv6_model(
    task_type, expected_key, lang_type
):
    model_key = resolve_model_key(
        task_type, OCRVersion.PPOCRV6, lang_type, ModelType.SMALL
    )

    assert model_key == expected_key


def test_indonesian_and_chinese_share_ppocrv6_model():
    indonesian = resolve_model_key(
        TaskType.REC, OCRVersion.PPOCRV6, "id", ModelType.SMALL
    )
    chinese = resolve_model_key(TaskType.REC, OCRVersion.PPOCRV6, "ch", ModelType.SMALL)

    assert indonesian == chinese == "multi_PP-OCRv6_rec_small"


def test_legacy_indonesian_rec_uses_latin_model_prefix():
    assert model_lang_prefix(TaskType.REC, OCRVersion.PPOCRV4, "id") == "latin"
    assert model_lang_prefix(TaskType.REC, OCRVersion.PPOCRV5, "indonesian") == "latin"
    assert model_lang_prefix(TaskType.REC, OCRVersion.PPOCRV5, "ch") == "ch"


def test_legacy_indonesian_det_uses_existing_detectors():
    assert model_lang_prefix(TaskType.DET, OCRVersion.PPOCRV4, "id") == "multi"
    assert model_lang_prefix(TaskType.DET, OCRVersion.PPOCRV5, "id") == "ch"


def test_indonesian_v5_rec_uses_latin_dictionary():
    model_info = InferSession.get_model_url(
        FileInfo(
            engine_type=EngineType.PADDLE,
            ocr_version=OCRVersion.PPOCRV5,
            task_type=TaskType.REC,
            lang_type="indonesian",
            model_type=ModelType.MOBILE,
        )
    )

    assert model_info["model_dir"].endswith("/latin_PP-OCRv5_rec_mobile")
    assert model_info["dict_url"].endswith("/ppocrv5_latin_dict.txt")


def test_chinese_v4_rec_model_is_unchanged():
    model_info = InferSession.get_model_url(
        FileInfo(
            engine_type=EngineType.ONNXRUNTIME,
            ocr_version=OCRVersion.PPOCRV4,
            task_type=TaskType.REC,
            lang_type=LangRec.CH,
            model_type=ModelType.MOBILE,
        )
    )

    assert model_info["model_dir"].endswith("/ch_PP-OCRv4_rec_mobile.onnx")
