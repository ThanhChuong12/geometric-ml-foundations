export interface Atom {
    atomicNumber: number;
    x: number;
    y: number;
    z: number;
}

export interface MoleculeInput {
    atomic_numbers: number[];
    positions: number[][];
}

export interface Molecule {
    name: string;
    atoms: Atom[];
}