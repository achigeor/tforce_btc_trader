"""
This file is for when you've found a solid hyper combo from hypersearch.py and you want to run it in the wild. Either
live, or "dry-run live" (--test-live), etc. Note, you need to run this file once first without live/test-live to
train and save the model (hypersearch doesn't save models).
"""

import argparse, os
from tensorforce.agents import agents as agents_dict
import shutil

import data
from btc_env import BitcoinEnv
from hypersearch import HSearchEnv

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_split', type=float, default=1, help="Num ways we'll split the GPU (how many tabs you running?)")
parser.add_argument('--id', type=int, help="Load winner from DB or hard-coded guess?")
parser.add_argument('--runs', type=int, default=15, help="Number of test-runs")
parser.add_argument('--live', action="store_true", default=False, help="Run in live mode")
parser.add_argument('--test-live', action="store_true", default=False, help="Dry-run live mode")
parser.add_argument('--early-stop', type=int, default=-1, help="Stop model after x successful runs")
parser.add_argument('--net-type', type=str, default='conv2d')  # todo pull this from winner automatically
parser.add_argument('--name', type=str, help="Name of the folder to save this run.")
args = parser.parse_args()


def main():
    directory = os.path.join(os.getcwd(), "saves", args.name)
    filestar = os.path.join(directory, args.name)

    live_ish = args.live or args.test_live
    if not live_ish:
        try: shutil.rmtree(directory)
        except: pass
        os.mkdir(directory)

    hs = HSearchEnv(gpu_split=args.gpu_split, net_type=args.net_type)
    flat, hydrated, network = hs.get_winner(id=args.id)
    env = BitcoinEnv(flat, name='ppo_agent')

    hydrated.update(
        # Agent
        states_preprocessing=None,
        actions_exploration=None,
        reward_preprocessing=None,
        # MemoryModel
        update_mode=dict(
            unit='episodes',
            # 10 episodes per update
            batch_size=20,
            # Every 10 episodes
            frequency=20
        ),
        memory=dict(
            type='latest',
            include_next_states=False,
            capacity=5000
        ),
        # DistributionModel
        distributions=None,
        # entropy_regularization=0.01,
        # PGModel
        # baseline_mode='states',
        # baseline=dict(
        #     type='mlp',
        #     sizes=[32, 32]
        # ),
        # baseline_optimizer=dict(
        #     type='multi_step',
        #     optimizer=dict(
        #         type='adam',
        #         learning_rate=1e-3
        #     ),
        #     num_steps=5
        # ),
        # gae_lambda=0.97,
        # PGLRModel
        # likelihood_ratio_clipping=0.2,
        # PPOAgent
        # step_optimizer=dict(
        #     type='adam',
        #     learning_rate=1e-3
        # ),
        subsampling_fraction=0.1,
        # optimization_steps=50
    )

    agent = agents_dict['ppo_agent'](
        states=env.states,
        actions=env.actions,
        network=network,
        **hydrated
    )

    if live_ish:
        agent.restore_model(directory)
        env.run_live(agent, test=args.test_live)
    else:
        env.train_and_test(agent, early_stop=args.early_stop, n_tests=args.runs)
        agent.save_model(filestar)
        agent.close()
        env.close()


if __name__ == '__main__':
    main()
