## Adversarial Discriminator

This directory contains the adversarial discriminator model meant to learn how to differentiate real from adversarial data. This model is meant to
test the stealthiness of our attack methods.

### Layout:
- `adversarialDiscriminator.py`: the PyTorch-Lightning module to train the model.
- `submodels.py`: the submodels of the discriminator module.
- `run_model.py`: script to run the adversarial discriminator model.