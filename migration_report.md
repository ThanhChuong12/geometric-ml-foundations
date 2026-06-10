# NequIP Configuration Migration Report (v0.7 to v0.18.0)

This report details the architectural changes required to migrate configuration files from the legacy NequIP v0.7 format to the modern NequIP v0.18.0 format (which utilizes Hydra and PyTorch Lightning).

## Key Migration Details

| Legacy Config Key (v0.7) | Modern Config Key (v0.18.0) | Reason for Change / Description |
| :--- | :--- | :--- |
| **`dataset_file_name`** | `data.split_dataset.file_path` | Transitioned to a PyTorch Lightning data module design. Dataloading configuration is nested under the `data` section with `_target_: nequip.data.datamodule.ASEDataModule`. |
| **`train_val_split`** / **`validation_split`** | `data.split_dataset.train`, `data.split_dataset.val`, `data.split_dataset.test` | Unified splitting ratios or counts are now encapsulated within the `split_dataset` section of the data module. |
| Implicit mapping | `data.transforms` | Data transformations (species-to-type mapping, neighbor list generation) are now explicit and modular transforms instantiated by Hydra (e.g. `ChemicalSpeciesToAtomTypeMapper`, `NeighborListTransform`). |
| **`batch_size`** / **`validation_batch_size`** | `data.train_dataloader.batch_size`, `data.val_dataloader.batch_size` | Dataloaders are standard `torch.utils.data.DataLoader` objects configured under the data module. |
| **`max_epochs`** | `trainer.max_epochs` | The PyTorch Lightning `Trainer` object manages the training loop. Max epochs is nested under the `trainer` block. |
| **`early_stopping_patience`** / **`early_stopping_lower_bounds`** | `trainer.callbacks` (specifically `lightning.pytorch.callbacks.EarlyStopping`) | Custom trainer patience logic has been replaced by standard PyTorch Lightning callbacks instantiated through Hydra. |
| **`loss_coeffs`** | `training_module.loss.coeffs` | The loss objective is configured using a subclass of PyTorch `nn.Module` (specifically `nequip.train.EnergyForceLoss`), permitting modular and custom losses. |
| **`learning_rate`** | `training_module.optimizer.lr` | Optimizers are instantiated as standard PyTorch optimizer classes (e.g. `torch.optim.Adam`) under the training module. |
| **`lr_decay_factor`** / **`lr_decay_patience`** | `training_module.lr_scheduler.scheduler` | Learning rate schedulers are now standard PyTorch schedulers (e.g. `ReduceLROnPlateau`) configured inside the training module. |
| Flat architecture keys (`num_layers`, `l_max`, `parity`, etc.) | `training_module.model` | Clean separation of model hyper-parameters from the training routine. Model variables are nested under the `training_module.model` section targeting `nequip.model.NequIPGNNModel`. |
| **`dataset_statistics`** (implicit) | `data.stats_manager` | Statistics collection (for shifts and scaling) is explicitly defined using the `stats_manager` subclass (e.g. `CommonDataStatisticsManager`). |

## Energy-Only Prediction Migration (QM9 Specific)

Since QM9 only contains relaxed equilibrium geometries where the net forces are physical zeros, the pipeline was migrated from a force-weighted loss to an energy-only target ($U_0$). Below is the mapping for this paradigm shift:

| Configuration Key | Force-based Setting | Modern Energy-Only Setting (v0.18.0) | Description |
| :--- | :--- | :--- | :--- |
| **`data.stats_manager._target_`** | `nequip.data.CommonDataStatisticsManager` | `nequip.data.EnergyOnlyDataStatisticsManager` | Avoids calculating force statistics (like `per_type_forces_rms`) and focuses only on energy statistics. |
| **`training_module.loss._target_`** | `nequip.train.EnergyForceLoss` | `nequip.train.EnergyOnlyLoss` | Removes force coefficient requirements; predicts only molecular energies. |
| **`training_module.val_metrics._target_`** | `nequip.train.EnergyOnlyMetrics` / `nequip.train.EnergyForceMetrics` | `nequip.train.EnergyOnlyMetrics` | Computes only energy metrics (`total_energy_mae: 1.0`), silencing force-related validation output. |
| **`training_module.model.do_derivatives`** | Implicitly `true` | `false` | Disables autograd gradient calculations with respect to coordinates, speeding up training and saving memory. |
| **`training_module.model.per_type_energy_scales`** | `${training_data_stats:per_type_forces_rms}` | `${training_data_stats:per_atom_energy_std}` | Mapped initialization energy scale to energy standard deviation instead of force RMS. |

## Validation & Compatibility

The updated configurations in [configs/](file:///d:/3rdY_HCMUS/Machine_Learning/LAB/L3/geometric-ml-foundations/ai_core/configs/) have been validated against the installed NequIP 0.18.0 package and PyTorch Lightning environment under WSL.

1. **Hydra Parsing**: Every configuration loads successfully with the Hydra CLI (`nequip-train --config-path <dir> --config-name <name>`).
2. **PyTorch Lightning Execution**: Checked CPU fallback training loop initialized and trained successfully over multiple epochs.
3. **Data Compatibility**: Successfully validated dataset loading, neighbor list building, statistics estimation, and model instantiation without forces.

