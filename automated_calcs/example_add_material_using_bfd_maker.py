from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad
from bfd_maker import BFDMaker  # <- make sure this is your class

# Set up LaunchPad
lp = LaunchPad.auto_load()

# Define base paths
vasp_dir = "../io_files/VASP_io/"         # Directory containing the POSCAR files
save_dir = "./"

# Initialize the Maker
maker = BFDMaker(directory=vasp_dir, save_dir=save_dir)

# List of materials to run
materials = [
    {"material_name": "Ba2CdAs2", "material_id": "mp-1079666"},
    {"material_name": "Sr2CdAs2", "material_id": "mp-867203"},
]

# Create and add each flow to LaunchPad
for mat in materials:
    flow = maker.make(material_name=mat["material_name"], material_id=mat["material_id"])
    lp.add_wf(flow_to_workflow(flow))
    print(f"Added {mat['material_name']} ({mat['material_id']}) to LaunchPad.")

