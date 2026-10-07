"""Tests for hydrogen-shift variant ions (c-1, z+1, z+2, ...) and the EAD models.

Ion keys: ``<series>[<signed variant>][<charge>]``. Variant 0 keeps the legacy
form (``z``, ``y2`` = doubly charged y). Non-zero variants separate the charge
with an underscore so it cannot be read as part of the variant: ``z+1``,
``z+1_2`` (doubly charged z+1), ``c-1``.
"""

import numpy as np
import pickle
import pytest
from ms2rescore_rs import (
    FragmentAnnotation,
    MS2Spectrum,
    Precursor,
    annotate_ms2_spectra,
    ms2pip_compute_theoretical_mz,
    ms2pip_extract_targets,
)

NH3 = 17.026549
H = 1.007825  # hydrogen atom
PROTON = 1.007276
TOL = 2e-3
LOG2_FLOOR = np.float32(np.log2(0.001))

PEPTIDE = "PEPTIDEK"
SEQ_LEN = len(PEPTIDE)


def _mz(ion_types, model, charge=2):
    return ms2pip_compute_theoretical_mz(
        [f"{PEPTIDE}/{charge}"], ion_types, model, "monoisotopic"
    )[0]


def _by():
    return _mz(["b", "y"], "cidhcd")


def _annotate(mz, model):
    spectrum = MS2Spectrum(
        identifier="s",
        mz=mz,
        intensity=[100.0] * len(mz),
        precursor=Precursor(mz=500.0, charge=2),
    )
    return annotate_ms2_spectra(
        spectra=[spectrum],
        proformas=[PEPTIDE],
        fragmentation_model=model,
        mass_mode="monoisotopic",
        tolerance_value=10.0,
        tolerance_mode="ppm",
    )[0]


class TestTheoreticalMzVariants:
    def test_etd_variant0_keys_use_variant0_mass(self):
        """'c' and 'z' must be c (= b+NH3) and z (= y-NH3), not whichever
        variant rustyms happens to generate first (c-1 / z-1 for ETD)."""
        by = _by()
        cz = _mz(["c", "z"], "etd")
        np.testing.assert_allclose(cz["c"], by["b"] + NH3, atol=TOL)
        np.testing.assert_allclose(cz["z"], by["y"] - NH3, atol=TOL)

    def test_ead_model_accepted(self):
        out = _mz(["b", "y", "c", "z", "z+1"], "ead")
        for key in ("b", "y", "c", "z", "z+1"):
            assert np.all(out[key] > 0), key

    def test_ead_ext_variants(self):
        by = _by()
        out = _mz(["c", "c-1", "z", "z+1", "z+2"], "ead_ext")
        np.testing.assert_allclose(out["c"], by["b"] + NH3, atol=TOL)
        np.testing.assert_allclose(out["c-1"], out["c"] - H, atol=TOL)
        np.testing.assert_allclose(out["z"], by["y"] - NH3, atol=TOL)
        np.testing.assert_allclose(out["z+1"], out["z"] + H, atol=TOL)
        np.testing.assert_allclose(out["z+2"], out["z"] + 2 * H, atol=TOL)

    def test_doubly_charged_variant_key(self):
        out = _mz(["z+1", "z+1_2"], "ead_ext", charge=3)
        np.testing.assert_allclose(out["z+1_2"], (out["z+1"] + PROTON) / 2, atol=TOL)

    def test_legacy_charge_key_unchanged(self):
        out = _mz(["y", "y2"], "cidhcd", charge=3)
        np.testing.assert_allclose(out["y2"], (out["y"] + PROTON) / 2, atol=TOL)

    def test_variant_key_not_aliased_to_charge(self):
        """'z+1' must select variant +1, not parse '+1' as charge 1 (= plain z)."""
        out = _mz(["z", "z+1"], "etd")
        np.testing.assert_allclose(out["z+1"], out["z"] + H, atol=TOL)

    @pytest.mark.parametrize("bad", ["z+1x", "z+_2", "q+1", "z+", "z1_2"])
    def test_invalid_ion_key_raises(self, bad):
        with pytest.raises(ValueError, match="ion type"):
            _mz([bad], "etd")


class TestFragmentAnnotationVariant:
    def test_default_variant_is_zero(self):
        assert FragmentAnnotation("z", 1, 1).variant == 0

    def test_variant_roundtrips_through_pickle(self):
        ann = FragmentAnnotation("z", 3, 2, variant=1)
        # In-memory round trip of an object made here (exercises __reduce__); no external data.
        back = pickle.loads(pickle.dumps(ann))
        assert (back.series, back.position, back.charge, back.variant) == ("z", 3, 2, 1)

    def test_annotation_reports_variant(self):
        z = _mz(["z", "z+1"], "ead")
        ann = _annotate([float(z["z"][0]), float(z["z+1"][0])], "ead")
        z_anns = [
            [(a.series, a.position, a.variant) for a in peak if a.series == "z"]
            for peak in ann.peak_annotations
        ]
        assert z_anns == [[("z", 1, 0)], [("z", 1, 1)]]


class TestExtractTargetsVariants:
    def test_z_variants_kept_apart(self):
        z = _mz(["z", "z+1"], "ead")
        ann = _annotate([float(z["z"][0]), float(z["z+1"][0])], "ead")
        targets = ms2pip_extract_targets(
            [ann], [np.array([0.5, 0.25], dtype=np.float32)], ["z", "z+1"], [SEQ_LEN]
        )[0]
        assert targets["z"][0] == pytest.approx(0.5)
        assert targets["z+1"][0] == pytest.approx(0.25)

    def test_neutral_loss_peak_not_counted_as_parent(self):
        """A y1-H2O peak must not become the y1 target."""
        y1 = float(_by()["y"][0])
        ann = _annotate([y1 - 18.010565], "ead")
        targets = ms2pip_extract_targets(
            [ann], [np.array([0.5], dtype=np.float32)], ["y"], [SEQ_LEN]
        )[0]
        assert targets["y"][0] == pytest.approx(LOG2_FLOOR)
