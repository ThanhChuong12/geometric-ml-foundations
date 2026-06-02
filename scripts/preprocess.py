import os
import sys
import argparse
import logging
import json
import random
import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Approximate atomic masses for common elements (fallback to atomic number z if not found)
ATOMIC_MASSES = {
    1: 1.008,    # H
    2: 4.0026,   # He
    3: 6.94,     # Li
    4: 9.0122,   # Be
    5: 10.81,    # B
    6: 12.011,   # C
    7: 14.007,   # N
    8: 15.999,   # O
    9: 18.998,   # F
    10: 20.180,  # Ne
    11: 22.990,  # Na
    12: 24.305,  # Mg
    13: 26.982,  # Al
    14: 28.085,  # Si
    15: 30.974,  # P
    16: 32.06,   # S
    17: 35.45,   # Cl
    18: 39.948,  # Ar
}

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to: {seed}")

def validate_sample(frame_idx, z, pos, energy, force):
    """
    Validate a single sample frame.
    Checks for empty arrays, shapes, NaNs, Infs, invalid atomic numbers,
    duplicate coordinate records, and shape mismatch.
    """
    # 1. Check for empty or missing inputs
    if z is None or len(z) == 0:
        logger.warning(f"Frame {frame_idx} rejected: atomic numbers array 'z' is empty or None.")
        return False
    if pos is None or len(pos) == 0:
        logger.warning(f"Frame {frame_idx} rejected: position array 'pos' is empty or None.")
        return False
    if force is None or len(force) == 0:
        logger.warning(f"Frame {frame_idx} rejected: force array 'force' is empty or None.")
        return False
    if energy is None:
        logger.warning(f"Frame {frame_idx} rejected: energy is None.")
        return False

    # 2. Check for NaN or Inf in any of the inputs
    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
        logger.warning(f"Frame {frame_idx} rejected: coordinates contain NaN or Inf.")
        return False
    if np.isnan(energy) or np.isinf(energy):
        logger.warning(f"Frame {frame_idx} rejected: energy contains NaN or Inf.")
        return False
    if np.any(np.isnan(force)) or np.any(np.isinf(force)):
        logger.warning(f"Frame {frame_idx} rejected: forces contain NaN or Inf.")
        return False

    # 3. Check shape dimensions
    num_atoms = len(z)
    if pos.ndim != 2 or pos.shape[1] != 3:
        logger.warning(f"Frame {frame_idx} rejected: position coordinate dimensions are incorrect (expected Nx3, got {pos.shape}).")
        return False
    if force.ndim != 2 or force.shape[1] != 3:
        logger.warning(f"Frame {frame_idx} rejected: force coordinate dimensions are incorrect (expected Nx3, got {force.shape}).")
        return False

    # 4. Check for inconsistent atom counts in this frame
    if pos.shape[0] != num_atoms:
        logger.warning(f"Frame {frame_idx} rejected: position row count {pos.shape[0]} does not match atomic numbers count {num_atoms}.")
        return False
    if force.shape[0] != num_atoms:
        logger.warning(f"Frame {frame_idx} rejected: force row count {force.shape[0]} does not match atomic numbers count {num_atoms}.")
        return False

    # 5. Check for invalid atomic numbers (z <= 0 or non-integer)
    if not np.issubdtype(z.dtype, np.integer):
        logger.warning(f"Frame {frame_idx} rejected: atomic numbers 'z' are not integers (dtype {z.dtype}).")
        return False
    if np.any(z <= 0):
        logger.warning(f"Frame {frame_idx} rejected: atomic numbers contain values <= 0.")
        return False

    # 6. Check for duplicated or malformed records (overlapping coordinates)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(num_atoms, dtype=bool)
    if np.any(dist[mask] < 1e-5):
        logger.warning(f"Frame {frame_idx} rejected: duplicate atom coordinates detected (distance < 1e-5).")
        return False

    return True

def center_positions(pos, z):
    """
    Translate atomic positions to place the center of mass at the origin.
    Uses atomic masses (or fallback to atomic numbers).
    """
    masses = np.array([ATOMIC_MASSES.get(int(zi), float(zi)) for zi in z]).reshape(-1, 1)
    total_mass = np.sum(masses)
    if total_mass <= 0:
        # Fallback to simple geometric center
        com = np.mean(pos, axis=0)
    else:
        com = np.sum(pos * masses, axis=0) / total_mass
    return pos - com

def compute_edge_index(pos, cutoff):
    """
    Compute neighborhood edge index list based on a distance cutoff.
    Returns a numpy array of shape [2, num_edges] representing directed edges.
    """
    num_atoms = pos.shape[0]
    if num_atoms <= 1:
        return np.empty((2, 0), dtype=np.int64)

    # Pairwise coordinate differences: [num_atoms, num_atoms, 3]
    diff = pos[:, None, :] - pos[None, :, :]
    # Pairwise Euclidean distances: [num_atoms, num_atoms]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # Check for non-finite edge construction results
    if np.any(np.isnan(dist)) or np.any(np.isinf(dist)):
        logger.error("Non-finite values encountered in pairwise distance calculation.")
        raise ValueError("Non-finite values encountered during edge construction.")
        
    # Create mask: distance <= cutoff and i != j (no self-loops)
    mask = (dist <= cutoff) & (~np.eye(num_atoms, dtype=bool))
    
    # edge_index is shape [2, num_edges]
    edge_index = np.argwhere(mask).T
    return edge_index

def build_graph_data(z, pos, energy, force, cutoff):
    """Construct PyTorch-ready dict representation of the graph with NequIP fields."""
    edge_index = compute_edge_index(pos, cutoff)
    
    # Wrap in PyTorch tensors
    data = {
        'atomic_numbers': torch.tensor(z, dtype=torch.long),
        'pos': torch.tensor(pos, dtype=torch.float32),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'total_energy': torch.tensor([energy], dtype=torch.float32),
        'forces': torch.tensor(force, dtype=torch.float32),
        'pbc': torch.tensor([False, False, False], dtype=torch.bool),
        'cell': torch.zeros((3, 3), dtype=torch.float32)
    }
    
    return data



def save_csv_long(z, pos_list, energy_list, force_list, indices, output_path):
    """Export dataset to a long-format CSV."""
    import csv
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'atom_idx', 'z', 'pos_x', 'pos_y', 'pos_z', 'force_x', 'force_y', 'force_z', 'energy'])
        for frame_idx in indices:
            pos = pos_list[frame_idx]
            force = force_list[frame_idx]
            energy = energy_list[frame_idx]
            for atom_idx, zi in enumerate(z):
                writer.writerow([
                    frame_idx,
                    atom_idx,
                    int(zi),
                    pos[atom_idx, 0],
                    pos[atom_idx, 1],
                    pos[atom_idx, 2],
                    force[atom_idx, 0],
                    force[atom_idx, 1],
                    force[atom_idx, 2],
                    energy
                ])

def save_csv_wide(z, pos_list, energy_list, force_list, indices, output_path):
    """Export dataset to a wide-format CSV (one row per frame)."""
    import csv
    num_atoms = len(z)
    header = ['frame_idx', 'energy']
    for idx in range(num_atoms):
        header.extend([f'pos_{idx}_x', f'pos_{idx}_y', f'pos_{idx}_z', f'z_{idx}', f'force_{idx}_x', f'force_{idx}_y', f'force_{idx}_z'])
        
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame_idx in indices:
            pos = pos_list[frame_idx]
            force = force_list[frame_idx]
            energy = energy_list[frame_idx]
            row = [frame_idx, energy]
            for atom_idx in range(num_atoms):
                row.extend([
                    pos[atom_idx, 0], pos[atom_idx, 1], pos[atom_idx, 2],
                    int(z[atom_idx]),
                    force[atom_idx, 0], force[atom_idx, 1], force[atom_idx, 2]
                ])
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Geometric ML Molecular Preprocessing Pipeline")
    parser.add_argument("--input_path", type=str, required=True, help="Path to raw .npz file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--format", type=str, default="both", choices=["pt", "csv", "both"], help="Output format: pt, csv, or both")
    parser.add_argument("--csv_style", type=str, default="long", choices=["long", "wide", "both"], help="CSV format style: long, wide, or both")
    parser.add_argument("--cutoff", type=float, default=4.0, help="Radial cutoff distance in Angstroms for graph creation")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training samples")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation samples")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--normalize", action="store_true", help="Apply standard scaling to energy and force targets")
    parser.add_argument("--force_scale_type", type=str, default="energy_scale", choices=["energy_scale", "force_std"],
                        help="How to scale forces: 'energy_scale' uses energy std (physically consistent), 'force_std' scales forces by their own std")
    parser.add_argument("--center", action="store_true", help="Center molecular coordinates at center-of-mass")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure file logging in output directory
    log_file = os.path.join(args.output_dir, "preprocess.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("Initializing Preprocessing Pipeline...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set seed
    set_seed(args.seed)
    
    # 1. Load Raw Data
    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        sys.exit(1)
        
    logger.info(f"Loading raw dataset from {args.input_path}...")
    try:
        raw_data = np.load(args.input_path)
    except Exception as e:
        logger.error(f"Failed to load .npz file: {e}")
        sys.exit(1)
        
    # Check for expected keys
    expected_keys = ['z', 'R', 'E', 'F']
    missing_keys = [k for k in expected_keys if k not in raw_data.files]
    if missing_keys:
        logger.error(f"Missing expected arrays in .npz file: {missing_keys}")
        sys.exit(1)
        
    z = raw_data['z']
    R = raw_data['R']
    E = raw_data['E']
    F = raw_data['F']
    
    n_frames = R.shape[0]
    num_atoms = len(z)
    logger.info(f"Raw dataset parsed successfully. Frames: {n_frames}, Atoms per frame: {num_atoms}")
    
    # 2. Validation & Cleaning
    valid_indices = []
    for i in range(n_frames):
        # E might be shape [N] or [N, 1]
        energy_val = E[i][0] if E.ndim > 1 and E.shape[1] == 1 else E[i]
        if validate_sample(i, z, R[i], energy_val, F[i]):
            valid_indices.append(i)
            
    n_valid = len(valid_indices)
    n_removed = n_frames - n_valid
    logger.info(f"Validation complete: {n_valid}/{n_frames} frames valid. Removed {n_removed} corrupted frames.")
    
    if n_valid == 0:
        logger.error("No valid frames remain after cleaning. Exiting.")
        sys.exit(1)
        
    # Normalize ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    train_ratio = args.train_ratio / total_ratio
    val_ratio = args.val_ratio / total_ratio
    
    # Shuffle and Split
    random.shuffle(valid_indices)
    train_cut = int(train_ratio * n_valid)
    val_cut = train_cut + int(val_ratio * n_valid)
    
    train_indices = valid_indices[:train_cut]
    val_indices = valid_indices[train_cut:val_cut]
    test_indices = valid_indices[val_cut:]
    
    logger.info(f"Dataset split indices created. Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Convert energy to 1D array of values for scaling calculation
    E_flat = np.array([E[i][0] if E.ndim > 1 else E[i] for i in range(n_frames)])
    
    # 3. Compute Normalization Parameters on Training Set
    energy_mean = 0.0
    energy_std = 1.0
    force_mean = 0.0
    force_std = 1.0
    
    if args.normalize:
        train_energies = E_flat[train_indices]
        train_forces = F[train_indices]
        
        energy_mean = float(np.mean(train_energies))
        energy_std = float(np.std(train_energies))
        if energy_std == 0:
            energy_std = 1.0
            
        if args.force_scale_type == "energy_scale":
            # Physically consistent force scaling: scale by energy scale factor
            force_mean = 0.0
            force_std = energy_std
        else:
            # Scale forces by their own standard deviation
            force_mean = float(np.mean(train_forces))
            force_std = float(np.std(train_forces))
            if force_std == 0:
                force_std = 1.0
                
        logger.info("Normalization parameters (computed on Train split):")
        logger.info(f"  Energy: mean = {energy_mean:.6f}, std = {energy_std:.6f}")
        logger.info(f"  Forces: mean = {force_mean:.6f}, std = {force_std:.6f} (Method: {args.force_scale_type})")
    
    # Prepare preprocessed coordinate lists and normalized target lists
    processed_pos = np.copy(R)
    processed_energy = np.copy(E_flat)
    processed_force = np.copy(F)
    
    # Apply centering if requested
    if args.center:
        logger.info("Centering coordinates at the center of mass for all frames...")
        for i in range(n_frames):
            processed_pos[i] = center_positions(processed_pos[i], z)
            
    # Apply normalization if requested
    if args.normalize:
        processed_energy = (processed_energy - energy_mean) / energy_std
        processed_force = (processed_force - force_mean) / force_std
        
    # 4. Export Outputs
    metadata = {
        'seed': args.seed,
        'cutoff': args.cutoff,
        'center_of_mass_centered': args.center,
        'normalized': args.normalize,
        'normalization': {
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'force_mean': force_mean,
            'force_std': force_std,
            'force_scale_type': args.force_scale_type
        },
        'num_atoms': num_atoms,
        'atomic_numbers': z.tolist(),
        'splits': {
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'total_valid': n_valid,
            'total_raw': n_frames
        }
    }
    
    # Save metadata JSON
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata summary exported to {metadata_path}")
    
    # Format exports
    if args.format in ["pt", "both"]:
        logger.info("Exporting PyTorch graph datasets (.pt)...")
        for split_name, split_idx in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
            split_data_list = []
            for idx in split_idx:
                graph = build_graph_data(
                    z=z,
                    pos=processed_pos[idx],
                    energy=processed_energy[idx],
                    force=processed_force[idx],
                    cutoff=args.cutoff
                )
                split_data_list.append(graph)
            
            output_pt_path = os.path.join(args.output_dir, f"{split_name}.pt")
            torch.save(split_data_list, output_pt_path)
            logger.info(f"  Saved {split_name}.pt (length: {len(split_data_list)})")
            
    if args.format in ["csv", "both"]:
        logger.info("Exporting CSV datasets (.csv)...")
        for split_name, split_idx in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
            if args.csv_style in ["long", "both"]:
                csv_path = os.path.join(args.output_dir, f"{split_name}_long.csv")
                save_csv_long(z, processed_pos, processed_energy, processed_force, split_idx, csv_path)
                logger.info(f"  Saved {split_name}_long.csv")
            if args.csv_style in ["wide", "both"]:
                csv_path = os.path.join(args.output_dir, f"{split_name}_wide.csv")
                save_csv_wide(z, processed_pos, processed_energy, processed_force, split_idx, csv_path)
                logger.info(f"  Saved {split_name}_wide.csv")
                
    logger.info("Preprocessing complete! All operations finished successfully.")

if __name__ == "__main__":
    main()
