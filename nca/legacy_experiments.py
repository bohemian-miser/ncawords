from nca.experiment import Experiment

class LegacyExperiment(Experiment):
    def __init__(self, id, title, dirname, desc, seed_type, c_n=16, h_n=80):
        self.legacy_id = id
        self.TITLE = title
        self.DIRNAME = dirname
        self.DESCRIPTION = desc
        self.SEED_TYPE = seed_type
        self.C_N = c_n
        self.H_N = h_n
        
    def get_metadata(self):
        return {
            "id": self.legacy_id,
            "title": self.TITLE,
            "dir": self.DIRNAME,
            "desc": self.DESCRIPTION,
            "seedType": self.SEED_TYPE,
            "c_n": self.C_N,
            "h_n": self.H_N
        }

class m1(LegacyExperiment):
    def __init__(self):
        super().__init__("m1", "Method 1: 3-Line BB", "snaps_web_method1/", "", "single", 32, 128)

class m1n(LegacyExperiment):
    def __init__(self):
        super().__init__("m1n", "Method 1: 3-Line BB (Noise)", "snaps_web_method1_noise/", "", "noise", 32, 128)

class m2(LegacyExperiment):
    def __init__(self):
        super().__init__("m2", "Method 2: Organic", "snaps_web_method2/", "", "single", 16, 80)

class m2n(LegacyExperiment):
    def __init__(self):
        super().__init__("m2n", "Method 2: Organic (Noise)", "snaps_web_method2_noise/", "", "noise", 16, 80)

class m4(LegacyExperiment):
    def __init__(self):
        super().__init__("m4", "Method 4: Proximity", "snaps_web_method4/", "", "single", 32, 128)

class m4n(LegacyExperiment):
    def __init__(self):
        super().__init__("m4n", "Method 4: Proximity (Noise)", "snaps_web_method4_noise/", "", "noise", 32, 128)

class m5(LegacyExperiment):
    def __init__(self):
        super().__init__("m5", "Method 5: Gravity", "snaps_web_method5/", "", "single", 32, 128)

class m5n(LegacyExperiment):
    def __init__(self):
        super().__init__("m5n", "Method 5: Gravity (Noise)", "snaps_web_method5_noise/", "", "noise", 32, 128)

class m9(LegacyExperiment):
    def __init__(self):
        super().__init__("m9", "9-Line Matrix", "snaps_9_line/", "WINNER", "single", 16, 80)

class m9n(LegacyExperiment):
    def __init__(self):
        super().__init__("m9n", "9-Line Matrix (Noise)", "snaps_9_line_noise/", "WINNER", "noise", 16, 80)

class evap(LegacyExperiment):
    def __init__(self):
        super().__init__("evap", "Evaporating Scaffold", "snaps_web_evaporate/", "", "single", 16, 80)

class evapn(LegacyExperiment):
    def __init__(self):
        super().__init__("evapn", "Evap Scaffold (Noise)", "snaps_web_evaporate_noise/", "", "noise", 16, 80)

class hid(LegacyExperiment):
    def __init__(self):
        super().__init__("hid", "Hidden Channel Scaffold", "snaps_web_hidden/", "", "single", 16, 80)

class hidn(LegacyExperiment):
    def __init__(self):
        super().__init__("hidn", "Hidden Channel (Noise)", "snaps_web_hidden_noise/", "", "noise", 16, 80)
