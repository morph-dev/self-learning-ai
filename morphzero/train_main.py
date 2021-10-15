import pickle
from datetime import datetime
from typing import Union

from morphzero.ai.algorithms.hash_policy import HashPolicy, StateHashPolicy, HashPolicyConfig
from morphzero.ai.algorithms.montecarlo import MonteCarloTreeSearch, MonteCarloTreeSearchConfig
from morphzero.ai.trainer import Trainer, TrainerConfig
from morphzero.core.game import Rules
from morphzero.games.connectfour.game import ConnectFourRules
from morphzero.games.genericgomoku.ai.tic_tac_toe import TicTacToeKeras, TicTacToeKerasConfig
from morphzero.games.genericgomoku.game import GenericGomokuRules


def short_str(
        config: Union[
            TrainerConfig,
            HashPolicyConfig,
            MonteCarloTreeSearchConfig,
            TicTacToeKerasConfig,
        ]) -> str:
    if isinstance(config, TrainerConfig):
        return f"__tr_i{config.iterations}_s{config.simulations}"
    if isinstance(config, HashPolicyConfig):
        return f"__hash_lr{config.learning_rate}_ex{config.exploration_rate}"
    if isinstance(config, MonteCarloTreeSearchConfig):
        return f"__mcts_sim{config.number_of_simulations}_ex{config.exploration_rate}_temp{config.temperature}"
    if isinstance(config, TicTacToeKerasConfig):
        return f"__keras_lr{config.learning_rate}_ep{config.epochs}_fil{config.filters}_midlayer{config.mid_layer_size}"
    raise ValueError("Unknown config type")


def train_hash_policy(
        rules: Rules,
        hash_policy_evaluator_config: HashPolicyConfig,
        trainer_config: TrainerConfig,
        path_prefix: str) -> None:
    model = HashPolicy(rules, StateHashPolicy(), hash_policy_evaluator_config)
    trainer = Trainer(rules, model, trainer_config)
    trainer.train()

    path = path_prefix + short_str(trainer_config) + short_str(hash_policy_evaluator_config)
    print(f"Storing at:\n{path}")
    model.policy.store(path)


def train_mcts_with_hash_policy(
        rules: Rules,
        mcts_config: MonteCarloTreeSearchConfig,
        hash_policy_config: HashPolicyConfig,
        trainer_config: TrainerConfig,
        path_prefix: str,
        store_training_data: bool) -> None:
    hash_policy = HashPolicy(rules, StateHashPolicy(), hash_policy_config)
    model = MonteCarloTreeSearch(rules, hash_policy, mcts_config)

    trainer = Trainer(rules, model, trainer_config)
    training_data = trainer.train()

    path = path_prefix + short_str(trainer_config) + short_str(hash_policy_config) + short_str(mcts_config)
    print(f"Storing HashPolicy at:\n{path}")
    hash_policy.policy.store(path)
    if store_training_data:
        with open(path + "_training_data", "wb") as f:
            pickle.dump(training_data, f)


def train_mcts_with_keras_tic_tac_toe(
        keras_config: TicTacToeKerasConfig,
        mcts_config: MonteCarloTreeSearchConfig,
        trainer_config: TrainerConfig,
        path: str) -> None:
    rules = GenericGomokuRules.create_tic_tac_toe_rules()
    keras_evaluator = TicTacToeKeras(keras_config)
    model = MonteCarloTreeSearch(rules, keras_evaluator, mcts_config)

    trainer = Trainer(rules, model, trainer_config)
    trainer.train()

    keras_evaluator.model.save(path)


def train() -> None:
    trainer_iterations = 1000
    trainer_simulations = 1

    hash_policy_learning_rate = 0.1
    hash_policy_exploration_rate = 0

    mcts_number_of_simulations = 1000
    mcts_exploration_rate = 1.4
    mcts_temperature = 1

    # # Train MCTS with Keras
    # train_mcts_with_keras_tic_tac_toe(
    #     TicTacToeKerasConfig(
    #         training=True,
    #         verbose=0,
    #         learning_rate=0.001,
    #         epochs=100,
    #         filters=100,
    #         mid_layer_size=100,
    #     ),
    #     MonteCarloTreeSearchConfig(
    #         number_of_simulations=mcts_number_of_simulations,
    #         exploration_rate=mcts_exploration_rate,
    #         temperature=mcts_temperature,
    #         max_time_sec=None,
    #     ),
    #     trainer_config=TrainerConfig(
    #         iterations=trainer_iterations,
    #         simulations=trainer_simulations,
    #     ),
    #     path="./../models/tic_tac_toe/keras_"
    # )

    # train MCTS with Hash Policy
    train_mcts_with_hash_policy(
        rules=ConnectFourRules.create_default_rules(),
        mcts_config=MonteCarloTreeSearchConfig(
            number_of_simulations=mcts_number_of_simulations,
            exploration_rate=mcts_exploration_rate,
            temperature=mcts_temperature,
            max_time_sec=None,
        ),
        hash_policy_config=HashPolicyConfig(
            learning_rate=hash_policy_learning_rate,
            exploration_rate=hash_policy_exploration_rate,
        ),
        trainer_config=TrainerConfig(
            iterations=trainer_iterations,
            simulations=trainer_simulations,
        ),
        path_prefix=f"./../models/connect4/hash_policy_mcts",
        store_training_data=True,
    )

    # # train hash_policy
    # train_hash_policy(
    #     rules=GenericGomokuRules.create_tic_tac_toe_rules(),
    #     hash_policy_evaluator_config=HashPolicyConfig(
    #         learning_rate=hash_policy_learning_rate,
    #         exploration_rate=hash_policy_exploration_rate,
    #     ),
    #     trainer_config=TrainerConfig(
    #         iterations=trainer_iterations,
    #         simulations=trainer_simulations,
    #     ),
    #     path_prefix=f"./../models/tic_tac_toe/hash_policy2",
    # )


if __name__ == "__main__":
    print("Start time:\t", datetime.now().strftime("%H:%M:%S"))
    train()
    print("End time:\t", datetime.now().strftime("%H:%M:%S"))
