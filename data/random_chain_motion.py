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
* `gemmi`  (conda‑forge or pip)
* `numpy`

    conda install -c conda-forge gemmi numpy

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
from pathlib import Path

import numpy as np
import gemmi

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


def transform_chain(chain: gemmi.Chain, rot: np.ndarray, trans: np.ndarray, pivot: np.ndarray) -> None:
    """Apply rotation *rot* and translation *trans* to every atom in *chain*."""
    for res in chain:
        for atom in res:
            v = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
            v_new = rot @ (v - pivot) + pivot + trans
            atom.pos.x, atom.pos.y, atom.pos.z = v_new


###############################################################################
# Hinge‑style domain motion
###############################################################################


def hinge_domain_motion(chain: gemmi.Chain, max_deg: float = 15.0) -> str:
    """Rotate the C‑terminal segment of *chain* around a backbone hinge.

    Returns a human‑readable description of the motion applied."""
    residues = [r for r in chain if r.has_atom("CA")]
    if len(residues) < 4:
        return "Chain too short for hinge motion"

    # Pick hinge residue roughly in middle ±25 % of chain length
    mid = len(residues) // 2
    quarter = len(residues) // 4
    hinge_idx = np.random.randint(max(1, mid - quarter), min(len(residues) - 2, mid + quarter))
    hinge_res = residues[hinge_idx]

    # Axis defined by CA of neighbors (i‑1 → i+1)
    p_prev = residues[hinge_idx - 1]["CA"].pos
    p_next = residues[hinge_idx + 1]["CA"].pos
    axis = np.array([p_next.x - p_prev.x,
                     p_next.y - p_prev.y,
                     p_next.z - p_prev.z])
    axis /= np.linalg.norm(axis)

    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

    pivot = np.array([hinge_res["CA"].pos.x,
                      hinge_res["CA"].pos.y,
                      hinge_res["CA"].pos.z])

    # Rotate all residues AFTER the hinge (i+1 ... end)
    for res in residues[hinge_idx + 1:]:
        for atom in res:
            v = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
            v_new = R @ (v - pivot) + pivot
            atom.pos.x, atom.pos.y, atom.pos.z = v_new

    angle_deg = abs(np.rad2deg(theta))
    return f"hinge at {hinge_res.seqid.num}{hinge_res.seqid.icode or ''}, angle {angle_deg:.1f}°"


###############################################################################
# Core processing per file
###############################################################################


def process_file(path: Path, outdir: Path, max_rot: float, max_trans: float) -> None:
    """Create two perturbed PDBs for *path* under *outdir*."""
    stem = path.stem

    # ---------------- RIGID‑BODY ----------------
    structure = gemmi.read_structure(str(path))
    model = structure[0]
    chains = [c for c in model]
    if not chains:
        print(f"[WARN] No chains found in {path.name}, skipping.")
        return

    chain = random.choice(chains)
    chain_id = chain.name
    coords = np.array([[atom.pos.x, atom.pos.y, atom.pos.z] for res in chain for atom in res])
    pivot = coords.mean(axis=0)
    rot = random_rotation_matrix(max_rot)
    trans = random_translation(max_trans)
    transform_chain(chain, rot, trans, pivot)

    out1 = outdir / f"{stem}_perturbed1.pdb"
    structure.write_pdb(str(out1))
    angle_rigid = math.acos(max(min((np.trace(rot) - 1) / 2, 1.0), -1.0))
    print(f"{path.name} → {out1.name}  (rigid, chain {chain_id}, rot {np.rad2deg(angle_rigid):.1f}°, trans {np.linalg.norm(trans):.2f} Å)")

    # ---------------- HINGE ----------------
    structure = gemmi.read_structure(str(path))  # fresh copy to avoid compounding
    model = structure[0]
    chains = [c for c in model]
    chain = random.choice(chains)
    chain_id = chain.name
    desc = hinge_domain_motion(chain, max_deg=max_rot)
    out2 = outdir / f"{stem}_perturbed2.pdb"
    structure.write_pdb(str(out2))
    print(f"{path.name} → {out2.name}  (hinge, chain {chain_id}, {desc})")


###############################################################################
# CLI entry‑point
###############################################################################


def main():
    parser = argparse.ArgumentParser(description="Apply rigid and hinge motions to random chains in mmCIF files, producing two PDBs each.")
    parser.add_argument("cifs", nargs="+", help="Input .cif /.mmcif files")
    parser.add_argument("--outdir", default="perturbed", help="Directory to write PDBs (default: %(default)s)")
    parser.add_argument("--max-rot", type=float, default=15.0, help="Maximum absolute rotation angle in degrees (default: %(default)s)")
    parser.add_argument("--max-trans", type=float, default=3.0, help="Maximum absolute translation in Å (rigid only, default: %(default)s)")
    args = parser.parse_args()

    # Optional reproducibility
    seed = os.getenv("MOTION_SEED")
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for cif in args.cifs:
        process_file(Path(cif), outdir, args.max_rot, args.max_trans)


if __name__ == "__main__":
    main()
