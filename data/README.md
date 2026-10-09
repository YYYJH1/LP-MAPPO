# Data

Energy is given in kJ and time in s.

| File | Content |
|---|---|
| `instances/params.json` | Scenario parameters of the instance generator |
| `results/test_episodes.csv.gz` | Per-episode test results (64,512 episodes) |
| `results/test_summary.csv` | Test results of all methods |
| `results/energy_components.csv` | Fleet energy per episode split into flight, computation, communication and payload energy |
| `results/training_curves.csv` | On-time completion fraction and feasibility rate during policy training |
| `results/scenario_pools.csv` | Results on the 15 scenario pools |
| `results/representative_episode.json` | One test episode of LP-MAPPO and CoPPO in detail |
| `results/flight_statistics.json` | Flight statistics of LP-MAPPO and CoPPO over all test episodes |
| `summaries/lp_mappo/test_summary.json` | Reference summary read by `scripts/summarize.py` |

## Test episodes

The test set has 256 instances, and each run is evaluated with two action seeds per instance.

- `method`, `policy`: evaluated method and its trained policy (LP-MAPPO uses MAPPO; empty for the reference methods).
- `setting`: `none` (no additional execution constraints), `feasibility` (feasibility constraints), `full` (plan time
  windows and feasibility constraints) or `reference`.
- `planner_seed` (`full` setting), `executor_seed`, `instance`, `action_seed`.
- `tasks`, `min_on_time`, `on_time_count`, `feasible`, `hard_violation`, `fleet_energy_kJ` and `loss` (the episode
  reward is `-loss`).

Energy per on-time task of a run, or of a combination of planner and policy runs, is its total fleet energy divided by
its total on-time count. `test_summary.csv` gives the means and standard errors over runs or run combinations.

## Scenario pools

Each pool changes one factor of the test setting: tasks per UAV, time-window scale or MEC capacity (4, 1 and
20 Gcycle/s in the test setting). The last column gives energy per on-time task relative to the best learning
baseline of the pool.
