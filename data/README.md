# Data

The LendingClub loan archive is not distributed in this repository.

To reproduce the analysis, obtain the accepted-loan archive used for this study and keep it locally as `archive.zip` or at another location. Pass the local path explicitly:

```powershell
python src/lendingclub_default_pipeline.py --archive "C:\path\to\archive.zip" --output-dir outputs\final_run --sample-rows 500000 --split both
```

Do not commit the source archive, the cleaned 500,000-loan modelling sample or row-level predictions. These files are excluded by `.gitignore`.
