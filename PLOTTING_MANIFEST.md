# Plotting Manifest

This manifest separates plotting scripts by their role in the current JINST-oriented paper plan. It is a planning and reproducibility guide only: diagnostic scripts are not deprecated or scheduled for deletion.

## Main-Paper Required

These scripts support the core detector/timing physics narrative and should remain reproducible for paper figures.

- `plot_timing_reconstruction_validation.py`
  Validates the reconstructed event timing against digi-level MC truth photon timing. Default truth timing is the amplitude-weighted mean of `RawCalorimeterHits/RawCalorimeterHits.timeStamp`; default photon count is the raw digi amplitude sum. Produces residual, timing-resolution, and waveform panels as PDF/PNG/JSON.

- `plot_depth_vs_q50.py`
  Shows the model-independent correlation between longitudinal shower depth and reconstructed timing `q50` for `e-`, `pi+`, `gamma`, and `pi0`. This is the minimal additional physics result connecting timing to shower interpretation. Produces PDF/PNG and optional JSON.

- `plot_shower_depth_distribution.py`
  Provides the longitudinal shower-depth comparison used to establish the physical separation between electromagnetic and hadronic shower development.

- `plot_performance_by_kinematics.py`
  Summarizes PID performance versus generated kinematics. Use this as an impact study after the timing and shower-observable figures establish detector meaning.

## Supporting

These scripts can be useful in the paper or appendix but should not define the main scientific claim.

- `plot_ablation_summary.py`
  Compares timing-feature controls and helps show that timing information carries measurable PID content.

- `plot_roc_curves.py`
  Produces class-pair ROC comparisons for model-performance context.

- `plot_confusion_matrix.py`
  Gives a compact PID classification summary.

- `plot_depth_time_2dhist.py`
  Visualizes hit-level depth-time density maps. Useful as supporting evidence for timing-depth structure.

- `plot_longitudinal_observables_summary.py`
  Collects several longitudinal shower observables in one publication-style figure. Useful when a broader shower-observable summary is needed.

- `plot_point_cloud_overview.py`
  Visualizes the hit-level point cloud distribution for an event. Essential for qualitative verification of shower structure and detector hit patterns.


## Diagnostic / Legacy

These scripts are kept for checks, debugging, and exploratory analysis. They are not required for the current minimum paper result.

- `plot_depth_time_event_examples.py`
- `plot_depth_binned_timing_spread_distribution.py`
- `plot_depth_time_residual_distribution.py`
- `plot_nearest_neighbor_distance_distribution.py`
- `plot_raw_charge_time.py`
- `plot_shower_properties.py`
- `plot_timing_quantiles.py`
- `plot_run_comparison.py`

When using a diagnostic plot in the manuscript, promote it explicitly by documenting the physical question it answers and adding it to the required or supporting list.
