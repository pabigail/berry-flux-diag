from dataclasses import dataclass
from pathlib import Path

from jobflow import Maker, Flow

from berry_flux_diag.BFDJobs import (
    preprocess_POSCARS,
    scf_with_fixed_kpoints,
    make_string_sum_jobs,
    calc_polarization_no_spin,
    bfd_schema,
)


@dataclass
class BFDMaker(Maker):
    """Maker to generate a BFD schema from relaxed POSCAR files.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    directory : str
        Where the relaxed POSCAR files are read from.
    save_dir : str
        Where the BFD schema JSON is written. Created if absent.

    Notes
    -----
    A Maker has to be a dataclass with a defaulted ``name`` field: jobflow
    inherits MSONable serialisation from it and reconstructs a maker from
    its dataclass fields, and atomate2 names jobs after ``name``. This
    class defined a plain ``__init__`` instead, so it had no fields to
    serialise and no name to report - it would not have survived a
    round trip through a job store.

    ``save_dir`` also defaulted to an absolute path on NERSC CFS, under a
    particular project and user, so the default was unusable for anyone
    else and silently wrong for the person it belonged to once they
    worked anywhere else. Both paths now default to somewhere relative to
    where the flow is built.
    """

    name: str = "BFD"
    directory: str = "."
    save_dir: str = "bfd_maker_outputs"

    def make(self, material_name: str, material_id: str):
        prefix = f"{material_id}_{material_name}"

        directory = Path(self.directory)
        save_dir = Path(self.save_dir)

        pol_poscar = directory / f"poscar_p_relaxed_{prefix}.vasp"
        np_poscar = directory / f"poscar_np_relaxed_{prefix}.vasp"

        # Ensure output directory exists
        save_dir.mkdir(parents=True, exist_ok=True)

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
            save_dir=str(save_dir),
        )

        # Return a Flow containing all jobs
        return Flow(
            [pre_process, scf_jobs, string_sum_jobs, calc_pol_job, save_data],
            name=f"{self.name}: {prefix}",
        )
