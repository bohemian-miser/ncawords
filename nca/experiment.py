import os
import json
from pathlib import Path

class Experiment:
    """
    Base class for all NCA training curriculum modules.
    Subclasses must define title, description, and target generation logic.
    """
    
    # Metadata to be exposed to the Dashboard API
    ID = "base_experiment"
    TITLE = "Base Experiment"
    DESCRIPTION = "Abstract base class."
    SEED_TYPE = "single"  # 'single' or 'noise'
    C_N = 16
    H_N = 80
    
    def __init__(self, base_dir="."):
        self.output_dir = Path(base_dir) / f"snaps_{self.ID}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_metadata(self):
        """Returns the dictionary format expected by the frontend UI."""
        return {
            "id": self.ID,
            "title": self.TITLE,
            "dir": f"snaps_{self.ID}/",
            "desc": self.DESCRIPTION,
            "seedType": self.SEED_TYPE,
            "c_n": self.C_N,
            "h_n": self.H_N
        }
        
    def generate_proposed_targets(self, total_steps: int = 4000):
        """
        Runs the mathematical/generative routine to design and write the 
        curriculum images `TARGET_{step:05d}.png` into the output directory.
        Allows the user to preview the environment dynamics before training.
        """
        raise NotImplementedError("Subclasses must implement target generation.")
        
    def train(self, total_steps: int = 4000):
        """
        The standardized PyTorch loop. It should load the targets generated
        by `generate_proposed_targets()` step-by-step and train the NCA model,
        outputting `COMP_{step:05d}.png` alongside the targets for the UI to read.
        """
        raise NotImplementedError("The base PyTorch tracking loop to be implemented.")

    @classmethod
    def register_all_methods(cls, output_file="methods.json"):
        """
        Discovers all subclasses and dumps their metadata into methods.json
        so the API/frontend dynamically picks them up on load.
        """
        methods = [subclass().get_metadata() for subclass in cls.__subclasses__()]
        with open(output_file, 'w') as f:
            json.dump(methods, f, indent=4)
        print(f"Exported {len(methods)} modular experiments to {output_file}")
