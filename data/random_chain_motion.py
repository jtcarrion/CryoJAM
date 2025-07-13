#!/usr/bin/env python3
"""
random_chain_motion.py ─ Augment protein coordinates with two flavors of motion
==========================================================================
For every input CIF/mmCIF file the script now produces **two** perturbed PDBs:

1. **Rigid‑body motion** applied to one randomly chosen chain → `*_perturbed1.pdb`
2. **Hinge‑style domain motion** applied to (a new) randomly chosen chain →
   `*_perturbed2.pdb`

Run twice? No need – both motions happen in a single pass, preserving the
original file in memory between the two operations.

Dependencies
------------
* `biopython`  (conda‑forge or pip)
* `numpy`

    conda install -c conda-forge biopython numpy

Usage examples
--------------
    # Default max ±15° rotation / ±3 Å translation, outputs next to originals
    python random_chain_motion.py  *.cif

    # Change motion ranges and output dir
    python random_chain_motion.py  --max-rot 30 --max-trans 5 --outdir aug  *.cif

    # Reproducible randomness
    MOTION_SEED=42 python random_chain_motion.py  *.cif
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO

###############################################################################
# Utility helpers
###############################################################################


def random_rotation_matrix(max_deg: float = 15.0) -> np.ndarray:
    """Generate a random 3×3 rotation matrix with angle ≤ *max_deg*."""
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def random_translation(max_ang: float = 3.0) -> np.ndarray:
    """Uniform random xyz shift within ±*max_ang* Å."""
    return np.random.uniform(-max_ang, max_ang, size=3)


def transform_chain(chain, rot: np.ndarray, trans: np.ndarray, pivot: np.ndarray) -> None:
    """Apply rotation *rot* and translation *trans* to every atom in *chain*."""
    for residue in chain:
        for atom in residue:
            v = np.array(atom.get_coord())
            v_new = rot @ (v - pivot) + pivot + trans
            atom.set_coord(v_new)


###############################################################################
# Hinge‑style domain motion
###############################################################################


def hinge_domain_motion(chain, max_deg: float = 15.0) -> str:
    """Rotate the C‑terminal segment of *chain* around a backbone hinge.

    Returns a human‑readable description of the motion applied."""
    # Get residues with CA atoms
    residues = [r for r in chain if 'CA' in r]
    if len(residues) < 4:
        return "Chain too short for hinge motion"

    # Pick hinge residue roughly in middle ±25 % of chain length
    mid = len(residues) // 2
    quarter = len(residues) // 4
    hinge_idx = np.random.randint(max(1, mid - quarter), min(len(residues) - 2, mid + quarter))
    hinge_res = residues[hinge_idx]

    # Axis defined by CA of neighbors (i‑1 → i+1)
    p_prev = residues[hinge_idx - 1]['CA'].get_coord()
    p_next = residues[hinge_idx + 1]['CA'].get_coord()
    axis = p_next - p_prev
    
    # Check if axis is valid (not zero length)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return "Invalid hinge axis (zero length)"
    
    axis /= axis_norm

    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

    pivot = np.array(hinge_res['CA'].get_coord())

    # Rotate all residues AFTER the hinge (i+1 ... end)
    for res in residues[hinge_idx + 1:]:
        for atom in res:
            v = np.array(atom.get_coord())
            v_new = R @ (v - pivot) + pivot
            atom.set_coord(v_new)

    angle_deg = abs(np.rad2deg(theta))
    return f"hinge at {hinge_res.get_id()[1]}, angle {angle_deg:.1f}°"


def complex_motion(structure, max_rot: float = 15.0, max_trans: float = 3.0) -> str:
    """Apply random motion to all chains in the structure.
    
    Returns a human‑readable description of the motions applied."""
    model = structure[0]
    chains = [c for c in model if len(c) > 0]
    
    if not chains:
        return "No valid chains found for complex motion"
    
    motion_descriptions = []
    
    for chain in chains:
        chain_id = chain.get_id()
        
        # Get all atom coordinates for pivot calculation
        coords = []
        for residue in chain:
            for atom in residue:
                coords.append(atom.get_coord())
        
        if not coords:
            continue
            
        coords = np.array(coords)
        pivot = coords.mean(axis=0)
        rot = random_rotation_matrix(max_rot)
        trans = random_translation(max_trans)
        transform_chain(chain, rot, trans, pivot)
        
        angle_rigid = math.acos(max(min((np.trace(rot) - 1) / 2, 1.0), -1.0))
        motion_descriptions.append(f"chain {chain_id}: rot {np.rad2deg(angle_rigid):.1f}°, trans {np.linalg.norm(trans):.2f} Å")
    
    return f"complex motion applied to {len(motion_descriptions)} chains: {'; '.join(motion_descriptions)}"


###############################################################################
# Core processing per file
###############################################################################


def process_file(path: Path, outdir: Path, max_rot: float, max_trans: float, complex_mode: bool = False) -> bool:
    """Create perturbed PDBs for *path* under *outdir*. Returns True if successful."""
    try:
        stem = path.stem

        # Choose parser based on file extension
        if path.suffix.lower() in ['.cif', '.mmcif']:
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        if complex_mode:
            # ---------------- COMPLEX MOTION ----------------
            structure = parser.get_structure('structure', str(path))
            if len(structure) == 0:
                print(f"[ERROR] No models found in {path.name}, skipping.")
                return False
                
            model = structure[0]
            chains = [c for c in model if len(c) > 0]
            if not chains:
                print(f"[ERROR] No valid chains found in {path.name}, skipping.")
                return False

            desc = complex_motion(structure, max_rot, max_trans)
            
            out_complex = outdir / f"{stem}_complex.pdb"
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(out_complex))
            print(f"{path.name} → {out_complex.name}  ({desc})")
            
            return True
        else:
            # ---------------- RIGID‑BODY ----------------
            structure = parser.get_structure('structure', str(path))
            if len(structure) == 0:
                print(f"[ERROR] No models found in {path.name}, skipping.")
                return False
                
            model = structure[0]
            chains = [c for c in model if len(c) > 0]  # Only chains with residues
            if not chains:
                print(f"[ERROR] No valid chains found in {path.name}, skipping.")
                return False

            chain = random.choice(chains)
            chain_id = chain.get_id()
            
            # Get all atom coordinates for pivot calculation
            coords = []
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
            
            if not coords:
                print(f"[ERROR] No atoms found in chain {chain_id} of {path.name}, skipping.")
                return False
                
            coords = np.array(coords)
            pivot = coords.mean(axis=0)
            rot = random_rotation_matrix(max_rot)
            trans = random_translation(max_trans)
            transform_chain(chain, rot, trans, pivot)

            out1 = outdir / f"{stem}_perturbed1.pdb"
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(out1))
            angle_rigid = math.acos(max(min((np.trace(rot) - 1) / 2, 1.0), -1.0))
            print(f"{path.name} → {out1.name}  (rigid, chain {chain_id}, rot {np.rad2deg(angle_rigid):.1f}°, trans {np.linalg.norm(trans):.2f} Å)")

            # ---------------- HINGE ----------------
            structure = parser.get_structure('structure', str(path))  # fresh copy to avoid compounding
            if len(structure) == 0:
                print(f"[ERROR] Failed to reload {path.name} for hinge motion, skipping.")
                return False
                
            model = structure[0]
            chains = [c for c in model if len(c) > 0]
            if not chains:
                print(f"[ERROR] No valid chains found in {path.name} for hinge motion, skipping.")
                return False
                
            chain = random.choice(chains)
            chain_id = chain.get_id()
            desc = hinge_domain_motion(chain, max_deg=max_rot)
            
            if "error" in desc.lower() or "invalid" in desc.lower():
                print(f"[WARNING] {path.name}: {desc}, skipping hinge motion.")
                return False
                
            out2 = outdir / f"{stem}_perturbed2.pdb"
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(out2))
            print(f"{path.name} → {out2.name}  (hinge, chain {chain_id}, {desc})")
            
            return True
        
    except Exception as e:
        print(f"[ERROR] Failed to process {path.name}: {str(e)}")
        return False


###############################################################################
# CLI entry‑point
###############################################################################


def main():
    parser = argparse.ArgumentParser(description="Apply rigid and hinge motions to random chains in mmCIF files, producing two PDBs each.")
    parser.add_argument("cifs", nargs="+", help="Input .cif /.mmcif files")
    parser.add_argument("--outdir", default="perturbed", help="Directory to write PDBs (default: %(default)s)")
    parser.add_argument("--max-rot", type=float, default=15.0, help="Maximum absolute rotation angle in degrees (default: %(default)s)")
    parser.add_argument("--max-trans", type=float, default=3.0, help="Maximum absolute translation in Å (rigid only, default: %(default)s)")
    parser.add_argument("--complex", action="store_true", help="Apply motion to all chains instead of individual chains")
    args = parser.parse_args()

    # Optional reproducibility
    seed = os.getenv("MOTION_SEED")
    if seed is not None:
        try:
            np.random.seed(int(seed))
            random.seed(int(seed))
            print(f"[INFO] Using seed: {seed}")
        except ValueError:
            print(f"[WARNING] Invalid seed '{seed}', using random seed.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Validate input files
    valid_files = []
    for cif_path in args.cifs:
        path = Path(cif_path)
        if not path.exists():
            print(f"[ERROR] File not found: {cif_path}")
            continue
        if not path.suffix.lower() in ['.cif', '.mmcif', '.pdb']:
            print(f"[WARNING] Unexpected file extension: {path.suffix}")
        valid_files.append(path)

    if not valid_files:
        print("[ERROR] No valid input files found.")
        sys.exit(1)

    print(f"[INFO] Processing {len(valid_files)} files...")
    if args.complex:
        print("[INFO] Using complex mode: applying motion to all chains")
    
    successful = 0
    for cif in valid_files:
        if process_file(cif, outdir, args.max_rot, args.max_trans, args.complex):
            successful += 1
    
    print(f"[INFO] Successfully processed {successful}/{len(valid_files)} files.")


if __name__ == "__main__":
    main()
