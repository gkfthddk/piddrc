# AI Agent Instructions for Dual-Readout PID Project

## 1. Python Environment (CRITICAL)
When running Python scripts or any ML-related code in this project, you **MUST** use the `torch251` mamba environment.

To activate and run code in this environment, prefix your execution with:
`mamba run -n torch251 <command>`
or
`source ~/.bashrc && mamba activate torch251 && <command>`

If you encounter `numpy.dtype size changed` or similar compiled library errors (e.g., from `h5py`), it is a strong indicator that you are using the default system Python instead of the `torch251` environment. Always ensure the environment is activated.

## 2. Generating Plots for the Paper
When creating or updating plots for the research paper (`main.tex`):
- Ensure **publication-quality aesthetics**: use large font sizes, clean visual styles (remove top/right spines), and optimized layouts for 2-column formatting.
- **Integrity**: Do NOT artificially exaggerate separation power between particle species (e.g., $\gamma$ vs $\pi^0$). Preserve realistic overlap (use transparency like `alpha=0.15` in scatter plots) and emphasize subtle, population-level differences.
- **Output**: Always export figures to both `.pdf` and `.png` formats (e.g., in `plots/publication/`).
- **Shared Utilities**: Utilize existing functions in `utils/plot_common.py` and `utils/plot_helpers.py` (e.g., `set_publication_style()`).

## 3. Data File Locations
The `.h5` / `.h5py` files are generally stored either locally in `h5s/` or centrally at `/store/ml/dual-readout/h5s/`. Use `resolve_paths` from `utils.plot_common` to handle file locating robustly.

## 3.1 Paper Direction and Analysis Defaults
For the current paper, use the following defaults unless the user explicitly overrides them:

- **Primary scientific framing**: this is a detector/timing physics study. The central claim is that high-resolution timing in a dual-readout calorimeter recovers longitudinal shower information. Mamba/ML performance is an analysis tool and validation channel, not the core scientific contribution.
- **Target style**: write for a JINST-style detector/instrumentation audience. Prioritize detector response, digitization, reconstruction stability, systematic limitations, and detector-design implications over generic ML benchmarking.
- **MC truth timing**: use digi-level `RawCalorimeterHits/RawCalorimeterHits.timeStamp` and define the event-level truth time as the amplitude-weighted mean of photon-arrival timestamps unless a section explicitly studies another estimator.
- **Photon count**: derive it from digi information and document the exact definition. Prefer raw amplitude sum when studying photon-statistics scaling; raw hit count is acceptable only when explicitly labeled as a count proxy.
- **Reco/truth matching**: match merged HDF5 events to digi ROOT events using HDF5 `seed` to locate the digi file and `E_gen` to identify the corresponding event within that file.
- **Main-text placement**: timing reconstruction validation belongs in the main manuscript, not only in an appendix or supplementary note.
- **Definitions must be explicit**: distinguish MC truth timing, reconstructed timing, photon count, waveform estimator, and any event-level timing summary (`t10`, `t50`, `t90`, RMS, etc.).

## 4. LaTeX Manuscript
When making changes to `main.tex`:
- Keep the academic tone consistent.
- Ensure proper referencing of newly generated PDF plots.
- Double-check that changes do not break the LaTeX compilation.
- Prioritize physics interpretation over model-centric narration. The paper should read as a detector/timing study using ML as one analysis tool, not as an ML benchmark paper.
- When adding or revising claims about timing, explicitly connect the chain:
  1. longitudinal shower development,
  2. optical photon propagation and digitized waveform structure,
  3. timing reconstruction stability,
  4. model-independent timing observables,
  5. downstream PID impact.
- For JINST-style writing, support claims with concrete detector/reconstruction quantities such as residual mean/RMS, photon-count dependence, waveform sampling assumptions, deconvolution behavior, and sensitivity to thresholds or timing smearing.
- Avoid overstating conclusions. Prefer language such as "indicates", "supports", "is consistent with", or "demonstrates within this simulation setup" unless the claim is directly and quantitatively established.
- Treat negative or small-gain results as informative physics constraints. For example, a small timing gain in `pi+` vs `e-` can support the interpretation that timing helps specifically where longitudinal ambiguity matters, rather than acting as a universal shortcut.

## 4.1 Scientific Value Checklist
When improving the manuscript, prefer additions that increase scientific interpretability, robustness, or detector-design relevance:

- **Truth and reconstruction definitions**: define what is used for MC truth timing, photon count, reconstructed timing, and waveform timing estimators. The default MC truth timing is the amplitude-weighted mean of digi `RawCalorimeterHits.timeStamp` after matching by `seed` and `E_gen`.
- **Timing reconstruction validation**: include residual mean/RMS, photon-count dependence, and waveform before/after Wiener deconvolution. Explain whether residual offsets are calibration constants, propagation offsets, or species-dependent physics effects.
- **Model-independent observables**: add or discuss simple timing quantities such as `t10`, `t50`, `t90`, `t90-t10`, timing RMS, late-light fraction, or depth-time correlations before relying on neural-network performance.
- **Orthogonality checks**: quantify whether timing adds information beyond `C/S`, total energy, shower shape, and geometry. Useful evidence includes timing-only baselines, no-timing baselines, shuffled-timing controls, integrated-waveform controls, or correlation studies.
- **Detector-design implications**: whenever possible, translate timing gains into requirements or sensitivities, such as timing smearing, waveform sampling interval, photon count, threshold, or deconvolution stability.
- **Phase-space dependence**: identify where timing helps and where it does not. Break down conclusions by energy, polar angle, shower overlap, or truth-level longitudinal structure.
- **Robustness**: check that conclusions survive reasonable changes in timing definition, event selection, photon-count threshold, and plotting range.

## 4.2 Preferred Logical Structure for Timing Claims
For any section claiming that timing improves PID, use this logic unless there is a strong reason not to:

1. Establish that the reconstructed timing observable is physically meaningful by comparing it to digi/MC truth timing, using amplitude-weighted `RawCalorimeterHits.timeStamp` as the default truth definition.
2. Show that timing correlates with longitudinal shower development in a model-independent way.
3. Show that the timing observable differs between the relevant physics classes, especially `gamma` and `pi0`.
4. Demonstrate that ML performance improves when timing is included.
5. Demonstrate through controls that the improvement comes from coherent time structure, not from energy, event ordering, or a global offset.
6. State the detector implication: what timing capability or waveform fidelity is needed for the effect to matter.

## 5. Available Helper Scripts (Skills)
Use these provided utility scripts to streamline tasks in the repository:

- **`./run_env.sh <command>`**
  A wrapper to execute any command safely within the required `torch251` environment without manually activating it.
  *Usage:* `./run_env.sh python plot_shower_depth_distribution.py`

- **`./compile.sh`**
  A LaTeX builder that automatically runs `pdflatex -> bibtex -> pdflatex -> pdflatex` on `main.tex` and halts on errors. Use this to verify your manuscript edits.
  *Usage:* `./compile.sh`

- **`./inspect_h5.py <path/to/file.h5>`**
  A lightweight inspector to quickly view the schema, groups, and dataset shapes of any HDF5 file.
  *Usage:* `./inspect_h5.py h5s/gamma_1-120GeV.h5py`

## 6. Complex Workflows (Skill Files)
For multi-step complex tasks, refer to the instruction files in the `skills/` directory. You can read these files using `view_file(IsSkillFile=True)` to follow a standardized procedure:

- **`skills/rebuild_paper.skill`**: Full pipeline to regenerate all publication figures and recompile the manuscript.
- **`skills/inspect_dataset.skill`**: Comprehensive check for newly added or modified datasets.


## 6. Faster Task Requests
If you want work to move faster and with fewer clarifications, give requests in a small structured block:

- Goal: what the figure/code/manuscript should achieve.
- Target files: exact scripts or sections to touch.
- Definitions: key physics choices that must be consistent.
  - Example: what counts as depth, what time definition to use, whether to use `time_end`.
- Output: filenames, formats, and destination paths.
- Allowed changes: whether helpers, refactors, or new files are acceptable.
- Stop condition: what counts as done.

Recommended request template:

```text
Goal: ...
Target files: ...
Definitions: ...
Output: ...
Allowed changes: ...
Stop condition: ...
```

For this project, the most useful early decision points are:

- depth definition: raw `z` vs shower-axis projected depth
- time definition: raw time, `time_end`, midpoint time, or event-shifted time
- granularity: event-level summary vs hit-level structure
- sample set: `e-`, `gamma`, `pi0`, `pi+`, or pairwise subsets
- style/output: PNG only vs PNG + PDF, single-panel vs multi-panel
