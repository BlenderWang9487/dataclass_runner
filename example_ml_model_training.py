from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from dataclass_runner import (
    BaseConfig,
    DataclassRunner,
    check_conflicting_params,
    runner_dataclass,
)

app = typer.Typer()


# define enum class (choices) for norm_type
# this can work with typer
class NormType(str, Enum):
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"


# define model config class
# model_name is a positional required field
@runner_dataclass
class ModelConfig(BaseConfig):
    model_name: str
    d_model: int = 24
    n_layers: int = 2
    norm_type: NormType = NormType.LAYER_NORM
    # you can even use Annotated to add help message, too
    n_heads: Annotated[int, typer.Option(help="head num of a AGI model")] = 4


# define train config class
# run_name, is a positional required field, too.
@runner_dataclass
class TrainConfig(BaseConfig):
    run_name: str
    # log_dir, is a required option (which must be specified with --log-dir "XXX")
    log_dir: Annotated[Path, typer.Option(help="log directory")]
    n_layers: int = 6
    batch_size: int = 32
    epochs: int = 10


@runner_dataclass
class RLTrainConfig(TrainConfig):
    exploration_rate: float = 0.1
    max_episode: int = 10


class Model:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.layers = [
            f"AGI forward pass {model_config.d_model}"
        ] * model_config.n_layers


class Trainer:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        self.log_file_name = train_config.log_dir / f"{train_config.run_name}.log"

    def train(self, model: Model):
        print(f"Training for {self.train_config.epochs} epochs")
        for i in range(self.train_config.epochs):
            print(f"Epoch {i + 1}: {model.layers}")
            print("model's layer:", model.layers)
        print(f"Log file: {self.log_file_name}")


class RLTrainer:
    def __init__(self, train_config: RLTrainConfig):
        self.train_config = train_config
        self.log_file_name = train_config.log_dir / f"{train_config.run_name}.log"

    def train(self, model: Model):
        print(f"Training for {self.train_config.epochs} epochs")
        for i in range(self.train_config.epochs):
            print(f"Epoch {i + 1}: {model.layers}")
            for e in range(self.train_config.max_episode):
                print(
                    f"Episode {e + 1}: Exploration rate {self.train_config.exploration_rate}"
                )
        print(f"Log file: {self.log_file_name}")


def main(args: "MainArgs"):
    # so this can be used with intellisense and auto-completion
    # which argparser cannot do
    # and compare to typer-only
    # and you don't need to handwrite all the variable assignment like this:
    #
    # def main(n_layers: int = 6, ...):
    #     model_config = ModelConfig(model_name, d_model, n_layers, norm_type)
    #     ...
    print(args)

    m = ModelConfig.from_config(args)
    print(m)

    ### this is equivalent, underlying it will convert instance that is BaseConfig to dict
    # m = ModelConfig.from_config(args.__dict__)

    t = TrainConfig.from_config(args)
    print(t)

    model = Model(m)
    trainer = Trainer(t)
    trainer.train(model)


# because runner_dataclass specify kw_only=True, so even if we have multiple
#   positional arguments in different parent class that don't have default values,
#   it will not break the __init__ method of Args
# check_conflicting_params is used to check if there are conflicting parameters in the
#   parent classes. It will raise an error if there are conflicting parameters. but
#   you can ignore some parameters (if you know what you are doing) by passing them in ignore_check_names
@DataclassRunner(app, callback=main)
@check_conflicting_params(ignore_check_names=["n_layers"])
@runner_dataclass
class MainArgs(TrainConfig, ModelConfig):
    pass


# demonstrate how to reuse the same dataclass for different runner
def rl_main(args: "RLMainArgs"):
    print(args)

    m = ModelConfig.from_config(args)
    print(m)

    t = RLTrainConfig.from_config(args)
    print(t)

    model = Model(m)
    trainer = RLTrainer(t)
    trainer.train(model)


@DataclassRunner(app, callback=rl_main)
@check_conflicting_params(ignore_check_names=["n_layers"])
@runner_dataclass
class RLMainArgs(RLTrainConfig, ModelConfig):
    pass


if __name__ == "__main__":
    # run:
    #    python ./example_ml_model_training.py --help
    # to see the help message of two commands (rl_main and main)

    # run:
    #    python ./example_ml_model_training.py main my_AGI_model run1 --log-dir ./logs --n-layers 4 --d-model 2
    # to see the output of main

    # run:
    #    python ./example_ml_model_training.py rl_main my_AGI_model run1 --log-dir ./logs --n-layers 4 --d-model 2 --exploration-rate 0.2 --epochs 2
    # to see the output of rl_main
    app()
