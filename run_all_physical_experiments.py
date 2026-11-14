"""
run_all_physical_experiments.py
Master runner that executes all 5 physical PDE + Split-Step Fourier BPM experiments
with full exception handling so the console window never closes unexpectedly.
"""
import traceback
from exp1_wavefront_conjugation import run_experiment_1
from exp2_particle_wake_pde import run_experiment_2
from exp3_chemical_kinetics_pde import run_experiment_3
from exp4_qubit_lindblad_pde import run_experiment_4
from exp5_continuous_xor_memory import run_experiment_5

if __name__ == "__main__":
    try:
        print("=======================================================================")
        print(" STARTING FULL PDE + SPLIT-STEP FOURIER BPM PHYSICAL BENCHMARK SUITE   ")
        print("=======================================================================\n")
        run_experiment_1()
        run_experiment_2()
        run_experiment_3()
        run_experiment_4()
        run_experiment_5()
        print("=======================================================================")
        print(" ALL 5 PHYSICAL EXPERIMENTS COMPLETED SUCCESSFULLY.                    ")
        print("=======================================================================")
    except Exception:
        print("\n[!] AN ERROR OCCURRED DURING EXECUTION:")
        traceback.print_exc()
    finally:
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass