import os
import sys
import argparse
import time

_SCRIPTS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import merge_gym
from _settings import Settings
import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    beam_plan,
    make_prefix,
    extract_actions,
    update_context,
)
import sumo

class Parser(utils.Parser):
    dataset: str = 'highway-v0'
    config: str = 'configs.offline'
    use_gui: bool = False
    num_episodes: int = 10
    max_steps: int = 500

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')

#######################
####### models ########
#######################

dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
                                 'data_config.pkl')

gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
                                  epoch=args.gpt_epoch, device=args.device)

#######################
####### dataset #######
#######################

if Settings.CUDA:
    device = "cuda"
else:
    device = "cpu"
Settings.USE_GUI = bool(args.use_gui)
env = merge_gym.GymEnvironment(Settings.GYM_ENVIRONMENT, device=device)
timer = utils.timer.Timer()

discretizer = dataset.discretizer
discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

value_fn = lambda x: discretizer.value_fn(x, args.percentile)
preprocess_fn = datasets.get_preprocess_fn(env.name)

#######################
###### main loop ######
#######################
all_episodes_planning_times = []
episode_summaries = []
step_records = []

for episode in range(args.num_episodes):
    print(f"Starting episode {episode + 1}")
    print("Resetting env...")
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    
    import numpy as np
    try:
        if isinstance(observation, dict):
            observation = observation.get('observation', observation)
        if hasattr(observation, 'observation'):
            observation = observation.observation
        if hasattr(observation, 'raw'):
            observation = observation.raw
        if hasattr(observation, 'cpu'):
            observation = observation.cpu().numpy()
        if hasattr(observation, 'numpy'):
            observation = observation.numpy()
        observation = np.array(observation).reshape(20, )
    except Exception as e:
        print(f"FATAL ERROR in observation extraction: {e}")
        raise e

    print("Env reset done.")
    total_reward = 0

    rollout = [observation.copy()]
    context = []

    T = args.max_steps
    current_episode_planning_times = []
    steps_run = 0

    for t in range(T):
        steps_run = t + 1

        observation = preprocess_fn(observation)

        if t % args.plan_freq == 0:
            print(f"Step {t} | starting plan...")
            prefix = make_prefix(discretizer, context, observation, args.prefix_context)

            start_plan_time = time.time()
            sequence = beam_plan(
                gpt, value_fn, prefix,
                args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                discount, args.max_context_transitions, verbose=args.verbose,
                k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
            )
            end_plan_time = time.time()
            plan_duration = end_plan_time - start_plan_time
            current_episode_planning_times.append(plan_duration)
            print(f"Step {t} | plan done in {plan_duration:.2f}s")
            step_plan_time = plan_duration

        else:
            sequence = sequence[1:]
            step_plan_time = None

        sequence_recon= discretizer.reconstruct(sequence)
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)
        planned_actions = extract_actions(sequence_recon, observation_dim, action_dim)

        step_result = env.step(action)
        if len(step_result) == 4:
            next_observation, reward, terminal, info = step_result
        else:
            next_observation, reward, terminated, truncated, info = step_result
            terminal = terminated or truncated

        import numpy as np
        try:
            if isinstance(next_observation, dict):
                next_observation = next_observation.get('observation', next_observation)
            if hasattr(next_observation, 'observation'):
                next_observation = next_observation.observation
            if hasattr(next_observation, 'raw'):
                next_observation = next_observation.raw
            if hasattr(next_observation, 'cpu'):
                next_observation = next_observation.cpu().numpy()
            if hasattr(next_observation, 'numpy'):
                next_observation = next_observation.numpy()
            next_observation = np.array(next_observation).reshape(20, )
        except Exception as e:
            print(f"FATAL ERROR in next_observation extraction: {e}")
            raise e

        total_reward += reward
        score = total_reward

        rollout.append(next_observation.copy())
        context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)

        try:
            action_value = float(np.array(action).reshape(-1)[0])
        except Exception:
            action_value = float(action)
        step_records.append({
            "episode": int(episode + 1),
            "step": int(t),
            "action": action_value,
            "reward": float(reward),
            "total_reward": float(total_reward),
            "planning_time_s": None if step_plan_time is None else float(step_plan_time),
            "planned_actions": planned_actions.tolist() if hasattr(planned_actions, "tolist") else planned_actions,
            "terminal": bool(terminal),
        })

        if terminal: break
        observation = next_observation

    avg_control = np.mean(env._env.unwrapped.control_history)
    avg_speed = np.mean(env._env.unwrapped.speed_history)
    avg_jerk = np.mean(env._env.unwrapped.jerk_history)
    u = env._env.unwrapped
    if hasattr(u, "_env"):
        u = u._env
    reward_steps = int(getattr(u, "_timing_steps", 0))
    reward_time_s = float(getattr(u, "_timing_reward_s", 0.0))
    state_time_s = float(getattr(u, "_timing_state_s", 0.0))
    vector_time_s = float(getattr(u, "_timing_vector_s", 0.0))

    if current_episode_planning_times:
        avg_plan_time_episode = np.mean(current_episode_planning_times)
        max_plan_time_episode = np.max(current_episode_planning_times)
        min_plan_time_episode = np.min(current_episode_planning_times)
        print(f"Episode {episode + 1} Planning Time (avg/max/min): {avg_plan_time_episode:.4f}s / {max_plan_time_episode:.4f}s / {min_plan_time_episode:.4f}s")
    else:
        avg_plan_time_episode = None
        max_plan_time_episode = None
        min_plan_time_episode = None
        print(f"Episode {episode + 1} No planning steps were timed.")

    all_episodes_planning_times.append(current_episode_planning_times)
    episode_summaries.append({
        "episode": int(episode + 1),
        "steps": int(steps_run),
        "total_reward": float(total_reward),
        "avg_control": float(avg_control),
        "avg_speed": float(avg_speed),
        "avg_jerk": float(avg_jerk),
        "reward_timing_steps": reward_steps,
        "reward_time_total_s": reward_time_s,
        "reward_time_avg_s": None if reward_steps == 0 else float(reward_time_s / reward_steps),
        "state_time_total_s": state_time_s,
        "state_time_avg_s": None if steps_run == 0 else float(state_time_s / steps_run),
        "vector_time_total_s": vector_time_s,
        "vector_time_avg_s": None if steps_run == 0 else float(vector_time_s / steps_run),
        "planning_calls": int(len(current_episode_planning_times)),
        "planning_time_avg_s": None if avg_plan_time_episode is None else float(avg_plan_time_episode),
        "planning_time_max_s": None if max_plan_time_episode is None else float(max_plan_time_episode),
        "planning_time_min_s": None if min_plan_time_episode is None else float(min_plan_time_episode),
    })

    print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")
    print(f"Average Control: {avg_control:.2f}")
    print(f"Average Speed: {avg_speed:.2f}")
    print(f"Average Jerk: {avg_jerk:.2f}\n")

print("\n--- Overall Planning Time Statistics ---")
all_planning_times_flat = [t for episode_times in all_episodes_planning_times for t in episode_times]

if all_planning_times_flat:
    overall_avg_plan_time = np.mean(all_planning_times_flat)
    overall_std_plan_time = np.std(all_planning_times_flat)
    overall_max_plan_time = np.max(all_planning_times_flat)
    overall_min_plan_time = np.min(all_planning_times_flat)
    overall_total_plan_time = np.sum(all_planning_times_flat)
    total_planning_calls = len(all_planning_times_flat)

    print(f"Total planning calls across {args.num_episodes} episodes: {total_planning_calls}")
    print(f"Average planning time per call: {overall_avg_plan_time:.4f}s")
    print(f"Standard deviation of planning time: {overall_std_plan_time:.4f}s")
    print(f"Maximum planning time in a single call: {overall_max_plan_time:.4f}s")
    print(f"Minimum planning time in a single call: {overall_min_plan_time:.4f}s")
    print(f"Total time spent on planning across all calls: {overall_total_plan_time:.2f}s")
else:
    print("No planning calls were made or timed across all episodes.")

export_payload = {
    "savepath": getattr(args, "savepath", None),
    "meta": {},
    "args": None,
    "episodes": episode_summaries,
    "steps": step_records,
    "overall": {
        "planning_calls": None if not all_planning_times_flat else int(total_planning_calls),
        "planning_time_avg_s": None if not all_planning_times_flat else float(overall_avg_plan_time),
        "planning_time_std_s": None if not all_planning_times_flat else float(overall_std_plan_time),
        "planning_time_max_s": None if not all_planning_times_flat else float(overall_max_plan_time),
        "planning_time_min_s": None if not all_planning_times_flat else float(overall_min_plan_time),
        "planning_time_total_s": None if not all_planning_times_flat else float(overall_total_plan_time),
    },
}

try:
    import json
    import csv
    from datetime import datetime, timezone
    export_payload["meta"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": getattr(args, "commit", None),
        "gpt_epoch": getattr(args, "gpt_epoch", None),
    }
    export_keys = [
        "dataset",
        "config",
        "logbase",
        "exp_name",
        "prefix",
        "suffix",
        "savepath",
        "device",
        "gpt_loadpath",
        "gpt_epoch",
        "plan_freq",
        "horizon",
        "beam_width",
        "n_expand",
        "k_obs",
        "k_act",
        "cdf_obs",
        "cdf_act",
        "percentile",
        "max_context_transitions",
        "prefix_context",
        "use_gui",
        "num_episodes",
        "max_steps",
    ]
    args_dict = {}
    for k in export_keys:
        if not hasattr(args, k):
            continue
        v = getattr(args, k)
        try:
            json.dumps(v)
            args_dict[k] = v
        except Exception:
            args_dict[k] = str(v)
    export_payload["args"] = args_dict
    savepath = getattr(args, "savepath", None)
    if savepath:
        json_path = os.path.join(savepath, "results.json")
        csv_path = os.path.join(savepath, "results.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, ensure_ascii=False, indent=2)
        if episode_summaries:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(episode_summaries[0].keys()))
                writer.writeheader()
                writer.writerows(episode_summaries)
        print(f"[ export ] Saved results: {json_path}")
        if episode_summaries:
            print(f"[ export ] Saved results: {csv_path}")
        if step_records:
            steps_csv_path = os.path.join(savepath, "steps.csv")
            with open(steps_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(step_records[0].keys()))
                writer.writeheader()
                writer.writerows(step_records)
            print(f"[ export ] Saved results: {steps_csv_path}")
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws1 = wb.active
            ws1.title = "episodes"
            if episode_summaries:
                ws1.append(list(episode_summaries[0].keys()))
                for row in episode_summaries:
                    ws1.append([row.get(k) for k in episode_summaries[0].keys()])
            ws2 = wb.create_sheet("steps")
            if step_records:
                ws2.append(list(step_records[0].keys()))
                for row in step_records:
                    ws2.append([row.get(k) for k in step_records[0].keys()])
            xlsx_path = os.path.join(savepath, "results.xlsx")
            wb.save(xlsx_path)
            print(f"[ export ] Saved results: {xlsx_path}")
        except Exception as e:
            print(f"[ export ] Excel export skipped: {e}")
except Exception as e:
    print(f"[ export ] Failed to save results: {e}")

sumo.exit_sumo()
