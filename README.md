# apex-bmp-climate-cbw
Code and data for study on agricultural best management practice (BMP) total nitrogen, total phosphorus, and sediment removal efficiencies will change under future climate scenarios (RCP 4.5 & 8.5) in the Chesapeake Bay Watershed, using [APEX](https://epicapex.tamu.edu) v1501 at field scale. 

Manuscript: in preparation for journal submission (link/DOI to be added on acceptance)

## what's here?

This repository contains the code, data, and derived results behind the paper's figures and tables. It does **not** contain the full APEX input/output archive (raw scenario files across all region x BMP x crop x climate regime combinations). A representative subset of raw APEX input/output files for one region/crop/BMP combination is included under 'data/example_apex_io/'.

## workflow
1. **`src/toolbox/`** - functions that edit APEX input files (.SOL, .OPC, .SUB), build/extend spin-up periods, apply climate perturbations, and process raw weather data into APEX-ready `.dly`/`.hly`/`.WP1` files. Code associated with running a variance-based (Sobol) sensitivity analysis also lives here.
2. **`src/scenario_build/`** - applies future climate to existing weather files, compiles discrete decadal climate runs into continuous 2021-2090 scenario files, grid-search parameter calibration, and the synthetic storm injection experiment.
3. **`src/extraction/`** - reads raw APEX output (SAD, ACY, DHY files) and builds the derived analysis datasets: annual results, two storm datasets (discrete and paired historical-to-future), an operation-phase-based dataset, and the Sobol samples.
4. **`src/analysis/`** - statistical tests (Mann-Kendall, Theil-Sen, Dunn's test, Spearman, bootstrap CIs) that produce statistical claims made in the text and compile SI tables.
5. **`src/plotting/`** - generates the main text and SI figures. `data/derived/` holds the output of stage 3 (the CSVs that stages 4 and 5 actually run on).

## data conventions
 
A few rules are applied consistently across the extraction and analysis scripts (consolidated in `src/toolbox/data_conventions.py`):
 
- MA (manure incorporation) is differenced against a dedicated `ma-base` baseline; every other BMP uses the shared `base` baseline.
- Rows with a crop failure (`YLDG + YLDF == 0`) are dropped, both for the row itself and its matched baseline.
- CC (cover crop) scenario SAD files carry two rows per day (cash + cover crop) with identical field-level loads; these are deduplicated to one row per day before summing.
- CC is only run for corn and soy; NT x alfalfa is not part of the study design.

## setup
 
```bash
pip install -r requirements.txt
```
 
Note: some early-stage scripts (`src/toolbox/`, `src/scenario_build/`) reference the raw APEX scenario file tree, which isn't included here (see above). Those scripts document the folder structure they expect via their configuration blocks at the top of each file; running them requires either the full archive (available by request) or adapting the paths to the example data.
 
## data availability
 
Derived datasets and analysis/plotting code are available in this repository, archived at Zenodo: [DOI to be added at submission].
Raw APEX scenario input/output files (full region x BMP x crop x climate matrix) are available from the corresponding author upon request, and are to be delivered directly to the Chesapeake Bay Program as project stakeholder.
 
## citation
 
Citation information will be added on publication.
