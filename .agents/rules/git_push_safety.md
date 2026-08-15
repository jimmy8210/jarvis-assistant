# Git Push Safety & Exclusions

- Always ensure large binary files (models, weights, .carpa, .fst, .raw), virtual environments (env/), logs (*.log), generated samples (xtts_samples/, *.wav), and sensitive configuration files (config.py, .env, pp_cache.json) are added to .gitignore.
- Before adding any new large asset or output directory, verify it is listed in .gitignore to prevent GitHub push failures or accidental key exposure.
