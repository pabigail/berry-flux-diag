"""Example: submit BFD flows for several materials to a FireWorks LaunchPad.

Not imported by the package - run it directly, after editing the two paths
below to point at your own POSCAR directory and output location.

Requires the VASP_atomate2 extra, a configured FireWorks LaunchPad, and
POSCAR files named as BFDMaker expects in `vasp_dir`.
"""

from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad

from berry_flux_diag.BFDMaker import BFDMaker

# Edit these: vasp_dir holds the input POSCAR files, save_dir receives the
# BFD schema output.
VASP_DIR = "../io_files/VASP_io/"
SAVE_DIR = "./"

# List of materials to run.
MATERIALS = [
    {"material_name": "Ba2CdAs2", "material_id": "mp-1079666"},
    {"material_name": "Sr2CdAs2", "material_id": "mp-867203"},
]


def main():
    lp = LaunchPad.auto_load()
    maker = BFDMaker(directory=VASP_DIR, save_dir=SAVE_DIR)

    for mat in MATERIALS:
        flow = maker.make(material_name=mat["material_name"],
                          material_id=mat["material_id"])
        lp.add_wf(flow_to_workflow(flow))
        print(f"Added {mat['material_name']} ({mat['material_id']}) to LaunchPad.")


if __name__ == "__main__":
    main()
