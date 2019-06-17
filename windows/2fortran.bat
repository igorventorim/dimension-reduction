rem ==================================================
rem Generating ..\te.exe ...
rem Run with: te ..\cfg\config_original_TE_default.csv
rem ==================================================

g77 -O3 -o ..\te -L. ..\te.f ..\teprob.f
