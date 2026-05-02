import numpy as np 

class LoopConfig:

    LOOP_DEFAULT = {
        "N_annotated": 500,
        "sampling_method": "random",
        "splits_ratio": [80, 10, 10],

        "model_name": "google-bert/bert-base-uncased",
        "n_epochs": 4, 
        "learning_rate": 3e-5,
        "weight_decay": 0,
        "batch_size": 8,

        "output_dir": "./models/current",
        "seed": 42,
        "device_batch_size": 4,
        "test_mode": False,
    }

    VARIABLES_TO_CHECK_FOR_EQUALITY = [
        "task_name", 
        "dichotomization_label", 

        "N_annotated", 
        "sampling_method", 
        "splits_ratio", 
        
        "model_name", 
        "n_epochs",
        "learning_rate", 
        "weight_decay", 
        "batch_size"
    ]

    def __init__(self, task_name : str, dichotomization_label : str, **kwargs) -> None:
        """
        Takes in any kwargs and return a dictionnary with the expected keys, default 
        values and format
        """
        self.task_name = str(task_name)
        self.dichotomization_label = str(dichotomization_label)

        self.N_annotated = int(
            kwargs.get("N_annotated", self.LOOP_DEFAULT["N_annotated"])
        )

        self.sampling_method = str(
            kwargs.get("sampling_method", self.LOOP_DEFAULT["sampling_method"])
        )
        
        self.splits_ratio = [int(v) for v in list(kwargs.get("splits_ratio", self.LOOP_DEFAULT["splits_ratio"]))]

        self.model_name = str(
            kwargs.get("model_name", self.LOOP_DEFAULT["model_name"])
        )
        
        self.n_epochs = int(
            kwargs.get("n_epochs", self.LOOP_DEFAULT["n_epochs"])
        )

        self.learning_rate = float(
            kwargs.get("learning_rate", self.LOOP_DEFAULT["learning_rate"])
        )
        
        self.weight_decay = float(
            kwargs.get("weight_decay", self.LOOP_DEFAULT["weight_decay"])
        )

        self.batch_size = int(
            kwargs.get("batch_size", self.LOOP_DEFAULT["batch_size"])
        )
        
        self.seed = int(
            kwargs.get("seed", self.LOOP_DEFAULT["seed"])
        )
        
        self.device_batch_size = int(
            kwargs.get("device_batch_size", self.LOOP_DEFAULT["device_batch_size"])
        )
        
        self.output_dir = str(
            kwargs.get("output_dir", self.LOOP_DEFAULT["output_dir"])
        )
        
        self.test_mode = bool(
            kwargs.get("test_mode", self.LOOP_DEFAULT["test_mode"])
        )

    def to_dict(self) -> dict:
        return {key : self.__getattribute__(key) for key in self.VARIABLES_TO_CHECK_FOR_EQUALITY}
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, LoopConfig):
            return TypeError("Can only check equality with LOOP_CONFIG objects")
        check_list = [
            self.__getattribute__(key) == __value.__getattribute__(key)
            for key in self.VARIABLES_TO_CHECK_FOR_EQUALITY
        ]
        return np.array(check_list).all()
