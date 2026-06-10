# backend/services/qm9_service.py

import ase
from nequip.data import AtomicDataDict
from nequip.data.ase import from_ase
from nequip.data.AtomicDataDict import with_batch_
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform
from typing import List


class MoleculeGraphService:
    """Service to convert molecular structures into NequIP-compatible graph dictionaries."""

    def __init__(self, type_names: List[str], r_max: float):
        self.type_names = type_names
        self.r_max = r_max

        # Instantiate transforms once to avoid initialization overhead during inference
        self.type_mapper = ChemicalSpeciesToAtomTypeMapper(
            model_type_names=self.type_names
        )
        self.neighbor_list = NeighborListTransform(r_max=self.r_max)

    def create_graph(
        self,
        atomic_numbers: List[int],
        positions: List[List[float]]
    ) -> AtomicDataDict.Type:
        """Create a single-molecule batched NequIP input graph dictionary.

        Args:
            atomic_numbers: List of integer atomic numbers.
            positions: List of 3D coordinate lists.

        Returns:
            A dictionary containing tensors formatted for NequIP model input.
        """
        # 1. Build an ASE Atoms object (non-periodic for isolated molecule)
        atoms = ase.Atoms(
            numbers=atomic_numbers,
            positions=positions,
            pbc=False
        )

        # 2. Convert ASE Atoms to base dictionary structure
        data = from_ase(atoms)

        # 3. Add single-frame batching fields in-place (batch mapping to 0)
        data = with_batch_(data)

        # 4. Apply transforms to map species to type indices and generate the neighbor list
        data = self.type_mapper(data)
        data = self.neighbor_list(data)

        return data