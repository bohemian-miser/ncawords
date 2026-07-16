import sys
from pathlib import Path

# Add project root to sys.path to allow nca imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from nca.experiment import Experiment

# Import all subclassed experiments here so they are registered in Experiment.__subclasses__()
try:
    from nca.train_guided import GuidedExperiment
except ImportError:
    pass

try:
    from nca.train_cloud import CloudExperiment
except ImportError:
    pass

try:
    from nca.train_dynamic_organic import DynamicOrganicExperiment
except ImportError:
    pass

def update_methods(output_file="methods.json"):
    """
    Orchestration hook: Generates the methods.json dynamically 
    based on all properly imported Experiment subclasses.
    """
    import json
    
    # Legacy methods that haven't been migrated to OOP yet
    legacy_methods = [
        { "id": "m1", "title": "Method 1: 3-Line BB", "dir": "snaps_web_method1/", "desc": "", "seedType": "single" },
        { "id": "m1n", "title": "Method 1: 3-Line BB (Noise)", "dir": "snaps_web_method1_noise/", "desc": "", "seedType": "noise" },
        { "id": "m2", "title": "Method 2: Organic", "dir": "snaps_web_method2/", "desc": "", "seedType": "single" },
        { "id": "m2n", "title": "Method 2: Organic (Noise)", "dir": "snaps_web_method2_noise/", "desc": "", "seedType": "noise" },
        { "id": "m4", "title": "Method 4: Proximity", "dir": "snaps_web_method4/", "desc": "", "seedType": "single" },
        { "id": "m4n", "title": "Method 4: Proximity (Noise)", "dir": "snaps_web_method4_noise/", "desc": "", "seedType": "noise" },
        { "id": "m5", "title": "Method 5: Gravity", "dir": "snaps_web_method5/", "desc": "", "seedType": "single" },
        { "id": "m5n", "title": "Method 5: Gravity (Noise)", "dir": "snaps_web_method5_noise/", "desc": "", "seedType": "noise" },
        { "id": "m9", "title": "9-Line Matrix", "dir": "snaps_9_line/", "desc": "WINNER", "seedType": "single" },
        { "id": "m9n", "title": "9-Line Matrix (Noise)", "dir": "snaps_9_line_noise/", "desc": "WINNER", "seedType": "noise" },
        { "id": "evap", "title": "Evaporating Scaffold", "dir": "snaps_web_evaporate/", "desc": "", "seedType": "single" },
        { "id": "evapn", "title": "Evap Scaffold (Noise)", "dir": "snaps_web_evaporate_noise/", "desc": "", "seedType": "noise" },
        { "id": "hid", "title": "Hidden Channel Scaffold", "dir": "snaps_web_hidden/", "desc": "", "seedType": "single" },
        { "id": "hidn", "title": "Hidden Channel (Noise)", "dir": "snaps_web_hidden_noise/", "desc": "", "seedType": "noise" }
    ]
    
    # base subclasses
    methods = legacy_methods + [subclass().get_metadata() for subclass in Experiment.__subclasses__()]
    
    # Generate the 12 parameterized versions of Dynamic Organic
    try:
        from nca.train_dynamic_organic import DynamicOrganicExperiment
        for n in [1, 100, 500, 1000]:
            for vol in [0.7, 0.4, 0.1]:
                vol_str = str(vol).replace('.', '_')
                exp = DynamicOrganicExperiment(
                    base_dir=".", 
                    text="COMP", 
                    update_every=n, 
                    support_vol=vol
                )
                exp.ID = f"dyn_clear_n{n}_vol{vol_str}"
                exp.TITLE = f"Dyn Organic N={n} Vol {vol}"
                exp.DESCRIPTION = f"Param grid config N={n}, vol={vol}"
                exp.output_dir = Path(f"snaps_{exp.ID}")
                
                # generate preview targets if they don't exist
                if not exp.output_dir.exists():
                    exp.generate_proposed_targets(total_steps=8000)
                    
                methods.append(exp.get_metadata())
    except ImportError:
        pass
        
    with open(output_file, 'w') as f:
        json.dump(methods, f, indent=4)
        
if __name__ == "__main__":
    update_methods()
