# -*- encoding: utf-8 -*-
from typing import List

import numpy as np
from rapidocr.ch_ppocr_cls import TextClsOutput
from rapidocr.ch_ppocr_rec import TextRecOutput
from rapidocr.main import apply_cls_flip_guard


def _rec(txts, scores, imgs=None, elapse=None) -> TextRecOutput:
    return TextRecOutput(
        imgs=imgs,
        txts=tuple(txts),
        scores=list(scores),
        word_results=tuple((txt, score, None) for txt, score in zip(txts, scores)),
        elapse=elapse,
    )


def test_flip_guard_keeps_higher_scoring_unflipped_crop():
    flipped = np.zeros((16, 80, 3), dtype=np.uint8)
    original = np.ones((16, 80, 3), dtype=np.uint8) * 255
    rec_res = _rec(["L0006Lh0h22Th0hE:"], [0.879], imgs=[flipped], elapse=0.02)
    cls_res = TextClsOutput(img_list=[flipped], cls_res=[("180", 0.99)])

    def recognize(imgs: List[np.ndarray]) -> TextRecOutput:
        assert len(imgs) == 1 and imgs[0] is original
        return _rec([":3404122404790007"], [0.996], imgs=imgs, elapse=0.01)

    new_rec, new_cls = apply_cls_flip_guard(
        rec_res, cls_res, [original], recognize, cls_thresh=0.9
    )

    assert new_rec.txts[0] == ":3404122404790007"
    assert new_rec.scores[0] == 0.996
    assert new_rec.imgs[0] is original
    assert new_cls.cls_res[0][0] == "0"
    assert new_cls.img_list[0] is original
    assert new_rec.elapse == 0.03


def test_flip_guard_keeps_flip_when_it_scores_higher():
    flipped = np.zeros((16, 80, 3), dtype=np.uint8)
    original = np.ones((16, 80, 3), dtype=np.uint8) * 255
    rec_res = _rec(["NIK 3404122404790007"], [0.97], imgs=[flipped])
    cls_res = TextClsOutput(img_list=[flipped], cls_res=[("180", 0.99)])

    def recognize(imgs: List[np.ndarray]) -> TextRecOutput:
        return _rec(["garbage"], [0.40], imgs=imgs)

    new_rec, new_cls = apply_cls_flip_guard(
        rec_res, cls_res, [original], recognize, cls_thresh=0.9
    )

    assert new_rec.txts[0] == "NIK 3404122404790007"
    assert new_rec.scores[0] == 0.97
    assert new_cls.cls_res[0][0] == "180"


def test_flip_guard_skips_when_classifier_did_not_rotate():
    crop = np.zeros((16, 80, 3), dtype=np.uint8)
    rec_res = _rec(["3404122404790007"], [0.99], imgs=[crop])
    cls_res = TextClsOutput(img_list=[crop], cls_res=[("0", 0.99)])
    called = []

    def recognize(imgs):
        called.append(True)
        raise AssertionError("should not re-recognize unflipped lines")

    new_rec, new_cls = apply_cls_flip_guard(
        rec_res, cls_res, [crop], recognize, cls_thresh=0.9
    )
    assert not called
    assert new_rec.txts[0] == "3404122404790007"
    assert new_cls.cls_res[0][0] == "0"
