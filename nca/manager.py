import sys
import json
from pathlib import Path
import inspect

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

try:
    import nca.legacy_experiments
except ImportError:
    pass

def update_methods(output_file="methods.json"):
    """
    Orchestration hook: Generates the methods.json dynamically 
    based on all properly imported Experiment subclasses.
    """
    
    def get_all_subclasses(cls):
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))
        return all_subclasses
        
    all_experiments = get_all_subclasses(Experiment)
    
    methods = []
    for cls in all_experiments:
        if inspect.isabstract(cls) or cls.__name__ in ('LegacyExperiment', 'DynamicOrganicExperiment'):
            continue
        try:
            methods.append(cls().get_metadata())
        except TypeError:
            pass # skip classes that can't be instantiated without arguments
            
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
