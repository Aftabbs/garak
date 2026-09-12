# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import importlib

from garak import _plugins
from garak import attempt
from garak.exception import GarakException
import garak.buffs.base

BUFFS = [classname for (classname, active) in _plugins.enumerate_plugins("buffs")]


@pytest.mark.parametrize("classname", BUFFS)
def test_buff_structure(classname):

    m = importlib.import_module("garak." + ".".join(classname.split(".")[:-1]))
    c = getattr(m, classname.split(".")[-1])

    # any parameter that has a default must be supported
    unsupported_defaults = []
    if c._supported_params is not None:
        if hasattr(c, "DEFAULT_PARAMS"):
            for k, _ in c.DEFAULT_PARAMS.items():
                if k not in c._supported_params:
                    unsupported_defaults.append(k)
    assert unsupported_defaults == []


@pytest.mark.parametrize("klassname", BUFFS)
def test_buff_load_and_transform(klassname, mocker):
    import sys

    try:
        b = _plugins.load_plugin(klassname)
    except GarakException:
        pytest.skip()
    assert isinstance(b, garak.buffs.base.Buff)
    a = attempt.Attempt()
    a.prompt = attempt.Message("I'm just a plain and simple tailor", lang=b.lang)

    if sys.platform == "win32" and klassname == "buffs.paraphrase.Fast":
        # special case buff not currently supported on Windows
        with pytest.raises(GarakException) as exc_info:
            list(b.transform(a))  # process yield to see raise
        assert "failed" in str(exc_info.value)
    else:
        # Model-backed buffs load a heavy seq2seq model on first use, but the
        # transform plumbing (dedup, attempt derivation, prompt rewrite) does not
        # depend on the generated text. Stub the model response so this stays a
        # unit test; real generation is covered by test_buff_results. Keyed on the
        # patched method itself so it tracks any buff that owns a _get_response.
        mocks_model = hasattr(b, "_get_response")
        if mocks_model:
            mocker.patch.object(
                b,
                "_get_response",
                return_value=["a paraphrase", "another paraphrase", "a paraphrase"],
            )
        buffed_a = list(b.transform(a))  # unroll the generator
        assert isinstance(buffed_a, list), "transform should return a list of attempts"
        if mocks_model:
            assert len(buffed_a) == 3, (
                "transform should yield the original attempt plus each unique "
                "paraphrase, with duplicates removed"
            )


def test_derive_new_attempt_deep_copies_mutable_fields():
    """_derive_new_attempt must not share mutable containers with the source attempt.

    Mutating notes, detector_results, targets, or probe_params on the derived
    attempt must not affect the source attempt (regression for GitHub #2155).
    """
    buff_instance = garak.buffs.base.Buff.__new__(garak.buffs.base.Buff)
    buff_instance.post_buff_hook = False
    buff_instance.fullname = "base.Buff"

    source = attempt.Attempt(
        probe_params={"key": "value", "nested": {"a": 1}},
        targets=["target1", "target2"],
        notes={"origin": "test", "nested_note": {"x": 42}},
        detector_results={"det.A": [0.1, 0.9]},
    )

    derived = buff_instance._derive_new_attempt(source)

    # Mutate every mutable field on the derived attempt
    derived.notes["injected"] = "should_not_appear_in_source"
    derived.notes["nested_note"]["x"] = 999
    derived.detector_results["det.A"].append(0.5)
    derived.detector_results["det.B"] = [0.0]
    derived.targets.append("target3")
    derived.probe_params["new_key"] = "new_val"
    derived.probe_params["nested"]["a"] = 99

    # Source must be unchanged
    assert "injected" not in source.notes, "source notes were mutated via derived attempt"
    assert source.notes["nested_note"]["x"] == 42, "source nested notes were mutated"
    assert len(source.detector_results["det.A"]) == 2, "source detector_results were mutated"
    assert "det.B" not in source.detector_results, "source detector_results gained a new key"
    assert len(source.targets) == 2, "source targets were mutated"
    assert "new_key" not in source.probe_params, "source probe_params were mutated"
    assert source.probe_params["nested"]["a"] == 1, "source nested probe_params were mutated"

    # The buff bookkeeping notes must only appear in derived, not source
    assert "buff_creator" in derived.notes
    assert "buff_creator" not in source.notes
