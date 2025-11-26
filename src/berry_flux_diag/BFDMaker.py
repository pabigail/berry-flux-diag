from pathlib import Path
from jobflow import Maker, Flow, job
from BFDJobs import (
    preprocess_POSCARS,
    scf_with_fixed_kpoints,
    make_string_sum_jobs,
    calc_polarization_no_spin,
    bfd_schema
)

class BFDMaker(Maker):
    """
    Maker to generate a BFD schema from relaxed POSCAR files.
    """

    def __init__(self, directory: str = "../", save_dir: str = "/global/cfs/cdirs/m4420/pabigail/atomate2_test/start_automating_bfd/bfd_maker_outputs/"):
        self.directory = Path(directory)
        self.save_dir = Path(save_dir)

    def make(self, material_name: str, material_id: str):
        prefix = f"{material_id}_{material_name}"
        pol_poscar = self.directory / f"poscar_p_relaxed_{prefix}.vasp"
        np_poscar = self.directory / f"poscar_np_relaxed_{prefix}.vasp"
        save_file = self.save_dir / f"bfd_schema_{prefix}.json"

        # Ensure output directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Call your jobs in sequence
        pre_process = preprocess_POSCARS(str(pol_poscar), str(np_poscar))
        scf_jobs = scf_with_fixed_kpoints(pre_process.output["structs"])
        string_sum_jobs = make_string_sum_jobs(scf_jobs.output.vasp_dirs)
        calc_pol_job = calc_polarization_no_spin(
            pre_process.output, string_sum_jobs.output["string_sum_outputs"]
        )
        save_data = bfd_schema(
            pre_process.output,
            scf_jobs.output,
            string_sum_jobs.output["string_sum_outputs"],
            calc_pol_job.output,
            save_name=prefix,
            save_dir=str(self.save_dir)
        )

        # Return a Flow containing all jobs
        return Flow(
            [pre_process, scf_jobs, string_sum_jobs, calc_pol_job, save_data],
            name=f"BFD: {prefix}"
        )
