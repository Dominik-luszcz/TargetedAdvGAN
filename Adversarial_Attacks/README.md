## Adversarial Attacks

This directory contains the baseline adversarial attacks in addition to the new slope attacks. Some code in this directory
is to generate results, so you may need to change some file paths, and these files may not be production level code.

### Layout:
- `AdversarialAttackClasses.py`: contains all adversarial attack classes including the new slope attacks.
- `slope_attacks.py`: script to run the slope attacks on the NHITS model.
- `baseline_attacks.py`: script to run the baseline attacks on the NHITS model.
- `data_extraction.py`: script to extract the test set recordings from the SP500.
- `epsilon_experiment.py`: script to run an epsilon experiment for each attack.