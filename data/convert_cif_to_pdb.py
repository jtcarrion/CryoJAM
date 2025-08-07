#!/usr/bin/env python3
"""
Convert CIF files to PDB format for compatibility with H5 generation scripts.
"""

import os
from pathlib import Path
from Bio import PDB
from Bio.PDB import MMCIFParser, PDBIO

def convert_cif_to_pdb(cif_file: Path, pdb_file: Path):
    """Convert a CIF file to PDB format."""
    parser = MMCIFParser()
    structure = parser.get_structure('structure', cif_file)
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_file))

def batch_convert_cif_to_pdb(cif_dir: Path, pdb_dir: Path):
    """Convert all CIF files in a directory to PDB format."""
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    cif_files = list(cif_dir.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files to convert...")
    
    for cif_file in cif_files:
        pdb_file = pdb_dir / f"{cif_file.stem}.pdb"
        print(f"Converting {cif_file.name} → {pdb_file.name}")
        try:
            convert_cif_to_pdb(cif_file, pdb_file)
        except Exception as e:
            print(f"Error converting {cif_file.name}: {e}")

if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Convert CIF files to PDB format")
    ap.add_argument("cif_dir", help="Directory containing CIF files")
    ap.add_argument("pdb_dir", help="Output directory for PDB files")
    
    args = ap.parse_args()
    
    batch_convert_cif_to_pdb(Path(args.cif_dir), Path(args.pdb_dir))
    print("Conversion complete!") 