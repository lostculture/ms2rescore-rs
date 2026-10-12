//! Internal fragment ions (yb, ya) for MS²PIP: segment enumeration, m/z and observed targets.

use std::collections::HashMap;

use rustyms::annotation::model::FragmentationModel;
use rustyms::chemistry::MassMode;
use rustyms::prelude::CompoundPeptidoformIon;

use crate::utils::build_theoretical_fragments;

pub const PROTON: f64 = 1.007_276_466_812;
pub const CO: f64 = 27.994_914_619_570;

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub s: usize,
    pub e: usize,
    pub n_isobaric: usize,
    pub mass: f64,
}

/// 1+ b-ion m/z by position (index k = b_k, k = 1..n-1; index 0 unused), from rustyms.
fn b_ladder(peptidoform: &CompoundPeptidoformIon, n: usize) -> Vec<f64> {
    let model = FragmentationModel::cid_hcd();
    let one = rustyms::system::isize::Charge::new::<rustyms::system::e>(1);
    let mut b = vec![f64::NAN; n];
    for f in build_theoretical_fragments(peptidoform, one, model, MassMode::Monoisotopic) {
        if f.series == 'b' && f.charge == 1 && f.variant == 0 && f.loss.is_empty() && f.position < n {
            b[f.position] = f.mz;
        }
    }
    b
}

/// Internal segments r(s)..r(e), 1 <= s < e <= n-2, residue sum < max_mass, in (s, e) order, with
/// isobaric segments (same residue composition) collapsed to the first. ProForma without charge suffix.
pub fn segments_for(proforma: &str, max_mass: f64) -> Result<Vec<Segment>, String> {
    let proforma = proforma.split_once('/').map_or(proforma, |(p, _)| p);
    let peptidoform = CompoundPeptidoformIon::pro_forma(proforma, None).map_err(|e| e.to_string())?;
    let n = peptidoform.peptidoforms().next().map(|p| p.sequence().len()).unwrap_or(0);
    if n < 4 {
        return Ok(Vec::new());
    }
    let b = b_ladder(&peptidoform, n);
    if b[1..].iter().any(|v| v.is_nan()) {
        return Err(format!("incomplete b ladder for {proforma}"));
    }
    let residue: Vec<f64> = (0..n - 1).map(|k| if k == 0 { f64::NAN } else { b[k + 1] - b[k] }).collect();
    let mut first: HashMap<Vec<i64>, usize> = HashMap::new();
    let mut out: Vec<Segment> = Vec::new();
    for s in 1..=n - 3 {
        for e in s + 1..=n - 2 {
            let mass = b[e + 1] - b[s];
            if mass >= max_mass {
                break; // residues are positive: longer segments from s are heavier
            }
            let mut key: Vec<i64> = (s..=e).map(|k| (residue[k] * 1e4).round() as i64).collect();
            key.sort_unstable();
            match first.get(&key) {
                Some(&i) => out[i].n_isobaric += 1,
                None => {
                    first.insert(key, out.len());
                    out.push(Segment { s, e, n_isobaric: 1, mass });
                }
            }
        }
    }
    Ok(out)
}


use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::annotation::parse_tolerance;
use crate::utils::{search_sorted_mz, LOG2_FLOOR_F64};

type FragmentArrays = (Py<PyArray1<i32>>, Py<PyArray1<i32>>, Py<PyArray1<i32>>, Py<PyArray1<f32>>, Py<PyArray1<f32>>);

/// Internal yb/ya fragments per PSM: (s, e, n_isobaric, mz_yb, mz_ya) in canonical segment order.
#[pyfunction]
pub fn ms2pip_internal_fragments(py: Python<'_>, proformas: Vec<String>, max_mass: f64) -> PyResult<Vec<FragmentArrays>> {
    let all: Result<Vec<Vec<Segment>>, String> =
        py.detach(|| proformas.par_iter().map(|p| segments_for(p, max_mass)).collect());
    let all = all.map_err(PyValueError::new_err)?;
    Ok(all
        .into_iter()
        .map(|segs| {
            let s = segs.iter().map(|g| g.s as i32).collect();
            let e = segs.iter().map(|g| g.e as i32).collect();
            let n = segs.iter().map(|g| g.n_isobaric as i32).collect();
            let yb = segs.iter().map(|g| (g.mass + PROTON) as f32).collect();
            let ya = segs.iter().map(|g| (g.mass - CO + PROTON) as f32).collect();
            (PyArray1::from_vec(py, s).into(), PyArray1::from_vec(py, e).into(), PyArray1::from_vec(py, n).into(),
             PyArray1::from_vec(py, yb).into(), PyArray1::from_vec(py, ya).into())
        })
        .collect())
}

/// Observed yb/ya intensities per PSM, in the order of `ms2pip_internal_fragments`; floor if unmatched.
#[pyfunction]
pub fn ms2pip_extract_internal_targets(
    py: Python<'_>,
    mzs: Vec<Py<PyArray1<f32>>>,
    intensities: Vec<Py<PyArray1<f32>>>,
    proformas: Vec<String>,
    max_mass: f64,
    tolerance_value: f64,
    tolerance_mode: String,
) -> PyResult<Vec<(Py<PyArray1<f32>>, Py<PyArray1<f32>>)>> {
    let n = proformas.len();
    if mzs.len() != n || intensities.len() != n {
        return Err(PyValueError::new_err("Input arrays must have identical length: mzs, intensities, proformas"));
    }
    let tolerance = parse_tolerance(tolerance_value, &tolerance_mode)?;
    let mut owned: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(n);
    for i in 0..n {
        let mz = unsafe { mzs[i].bind(py).as_slice()? }.to_vec();
        let it = unsafe { intensities[i].bind(py).as_slice()? }.to_vec();
        if mz.len() != it.len() {
            return Err(PyValueError::new_err(format!("Spectrum {i}: mz/intensity length mismatch")));
        }
        if mz.windows(2).any(|w| w[1] < w[0]) {
            return Err(PyValueError::new_err(format!("Spectrum {i}: m/z must be sorted ascending")));
        }
        owned.push((mz, it));
    }
    let floor = LOG2_FLOOR_F64 as f32;
    let out: Result<Vec<(Vec<f32>, Vec<f32>)>, String> = py.detach(|| {
        owned
            .par_iter()
            .zip(proformas.par_iter())
            .map(|((mz, it), pf)| {
                let segs = segments_for(pf, max_mass)?;
                let pick = |target: f64| match search_sorted_mz(mz, target, &tolerance) {
                    Some(i) => it[i],
                    None => floor,
                };
                Ok((
                    segs.iter().map(|g| pick(g.mass + PROTON)).collect(),
                    segs.iter().map(|g| pick(g.mass - CO + PROTON)).collect(),
                ))
            })
            .collect()
    });
    let out = out.map_err(PyValueError::new_err)?;
    Ok(out
        .into_iter()
        .map(|(yb, ya)| (PyArray1::from_vec(py, yb).into(), PyArray1::from_vec(py, ya).into()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    const P: f64 = 97.052_764; const E: f64 = 129.042_593; const A: f64 = 71.037_114;
    const G: f64 = 57.021_464; const M: f64 = 131.040_485; const OX: f64 = 15.994_915;

    #[test]
    fn segment_order_and_mass() {
        let segs = segments_for("PEPTIDEK", 700.0).unwrap();
        assert_eq!((segs[0].s, segs[0].e), (1, 2));
        assert!((segs[0].mass - (E + P)).abs() < 1e-3);
        let mut sorted = segs.clone();
        sorted.sort_by_key(|g| (g.s, g.e));
        assert_eq!(segs, sorted);
        assert!(segs.iter().all(|g| g.e <= 6 && g.s >= 1 && g.e > g.s && g.mass < 700.0));
    }

    #[test]
    fn isobaric_segments_collapse() {
        let segs = segments_for("PAGAGK", 700.0).unwrap();
        let got: Vec<(usize, usize, usize)> = segs.iter().map(|g| (g.s, g.e, g.n_isobaric)).collect();
        assert_eq!(got, vec![(1, 2, 3), (1, 3, 1), (1, 4, 1), (2, 4, 1)]);
        assert!((segs[0].mass - (A + G)).abs() < 1e-3);
    }

    #[test]
    fn internal_masses_include_modifications() {
        let plain = segments_for("PMAGK", 700.0).unwrap();
        let ox = segments_for("PM[+15.9949]AGK", 700.0).unwrap();
        assert!((plain[0].mass - (M + A)).abs() < 1e-3);
        assert!((ox[0].mass - (M + OX + A)).abs() < 1e-3);
        let ac = segments_for("[+42.0106]-PMAGK", 700.0).unwrap();
        assert!((ac[0].mass - plain[0].mass).abs() < 1e-6);
    }

    #[test]
    fn mass_cut_and_short_peptides() {
        assert!(segments_for("PEPK", 700.0).unwrap().len() == 1);
        assert!(segments_for("PEK", 700.0).unwrap().is_empty());
        assert!(segments_for("PEPTIDEK", 150.0).unwrap().is_empty());
    }
}
