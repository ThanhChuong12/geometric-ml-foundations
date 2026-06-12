import ase
from nequip.data import AtomicDataDict
from nequip.data.ase import from_ase
from nequip.data.AtomicDataDict import with_batch_
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform
from typing import List


class MoleculeGraphService:
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
        # Build an ASE Atoms object (non-periodic for isolated molecule)
        atoms = ase.Atoms(
            numbers=atomic_numbers,
            positions=positions,
            pbc=False
        )

        # Convert ASE Atoms to base dictionary structure
        data = from_ase(atoms)

        # Add single-frame batching fields in-place (batch mapping to 0)
        data = with_batch_(data)

        # Apply transforms to map species to type indices and generate the neighbor list
        data = self.type_mapper(data)
        data = self.neighbor_list(data)

        return data