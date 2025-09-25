## Stock Projection Model

This directory contains the NHITS (from pytorch-forecasting) model for forecasting stock adjprc. Note that we use the adjprc version of the model, given that forecasting returns gives sub-par results. 

### Layout:
- `preprocessing_and_data_collection`: Contains preprocessing code in addition to data collection from WRDS.
- `forecasting_model_adjprc.py`: the NHITS model for forecasting adjprc.
- `forecasting_model_ret.py`: the NHITS model for forecasting returns (not used).