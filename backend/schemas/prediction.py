from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List

class MoleculeInput(BaseModel):
    atomic_numbers: List[int] = Field(
        ..., 
        description="List of atomic numbers (elements) of the atoms in the molecule."
    )
    positions: List[List[float]] = Field(
        ..., 
        description="List of 3D coordinates [x, y, z] for each atom."
    )

    @field_validator("atomic_numbers")
    @classmethod
    def validate_atomic_numbers(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("atomic_numbers list must not be empty.")
        for idx, z in enumerate(v):
            if z <= 0:
                raise ValueError(
                    f"Invalid atomic number {z} at index {idx}. "
                    "Atomic numbers must be positive integers."
                )
        return v

    @field_validator("positions")
    @classmethod
    def validate_positions(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("positions list must not be empty.")
        for idx, pos in enumerate(v):
            if len(pos) != 3:
                raise ValueError(
                    f"Position at index {idx} must have exactly 3 coordinates [x, y, z], "
                    f"got {len(pos)}."
                )
        return v

    @model_validator(mode="after")
    def validate_lengths_match(self) -> "MoleculeInput":
        len_z = len(self.atomic_numbers)
        len_pos = len(self.positions)
        if len_z != len_pos:
            raise ValueError(
                f"Mismatch between the number of atomic numbers ({len_z}) "
                f"and positions ({len_pos})."
            )
        return self


class EnergyResponse(BaseModel):
    energy: float = Field(
        ..., 
        description="Predicted internal molecular energy (U0) in eV."
    )