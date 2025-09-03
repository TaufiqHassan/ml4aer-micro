## Script to generate E3SM training data

* Generates training/validation/testing data for ML model in npy format directly from E3SM outputs.
* Generate 2D historgrams of the data for distribution visualization.

Usage
-----

```console
python extract_train_info.py -h
```
```bash
usage: extract_train_info.py [-h] --datadir DATADIR --data_files DATA_FILES [DATA_FILES ...]
                             [--timeslice TIMESLICE TIMESLICE] [--levslice LEVSLICE LEVSLICE] [--varlist [VARLIST ...]]
                             [--atmos_vars [ATMOS_VARS ...]] [--cloudfree CLOUDFREE] [--outdir OUTDIR] [--out OUT]
                             [--test_size TEST_SIZE] [--logscale LOGSCALE] [--endstring ENDSTRING] --action
                             {gen_data,gen_hist,gen_all_hist}

Generate training data and plot distribution histograms for E3SM data.

options:
  -h, --help            show this help message and exit
  --datadir DATADIR     Directory where the data files are located.
  --data_files DATA_FILES [DATA_FILES ...]
                        List of data file names.
  --timeslice TIMESLICE TIMESLICE
                        Time slice as a list of two integers (start, end).
  --levslice LEVSLICE LEVSLICE
                        Level slice as a list of two integers (start, end).
  --varlist [VARLIST ...]
                        List of mixing ratio variables.
  --atmos_vars [ATMOS_VARS ...]
                        List of atmospheric condition variables.
  --cloudfree CLOUDFREE
                        Whether to mask out cloudy grids (default: True).
  --outdir OUTDIR       Output directory to save numpy arrays and plots.
  --out OUT             Whether to save the output files (default: True).
  --test_size TEST_SIZE
                        Test set size as a proportion (default: 0.66).
  --logscale LOGSCALE   Whether to use log scale in the plots (default: False).
  --endstring ENDSTRING
                        Suffix for the output file names.
  --action {gen_data,gen_hist,gen_all_hist}
                        Choose an action: 'gen_data' to generate training data or 'get_hist' to generate 2D histograms or
                        'gen_all_hist' to generate two 2D histograms for state variables and tendencies.
```
