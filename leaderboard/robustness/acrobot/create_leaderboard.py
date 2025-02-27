import os
import importlib
import argparse
import pandas
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters

# from double_pendulum.analysis.leaderboard_utils import compute_leaderboard
from double_pendulum.analysis.leaderboard import leaderboard_scores
from double_pendulum.analysis.benchmark_scores import get_scores

from sim_parameters import mpar, dt, t_final, t0, x0, goal, integrator
from benchmark_controller import benchmark_controller


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    dest="data_dir",
    help="Directory for saving data. Existing data will be kept.",
    default="data",
    required=False,
)
parser.add_argument(
    "--force-recompute",
    dest="recompute",
    help="Whether to force the recomputation of the leaderboard even without new data.",
    default=False,
    required=False,
    type=bool,
)

parser.add_argument(
    "--link-base",
    dest="link",
    help="base-link for hosting data. Not needed for local execution",
    default="",
    required=False,
)

data_dir = parser.parse_args().data_dir
recompute_leaderboard = parser.parse_args().recompute
link_base = parser.parse_args().link

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

existing_list = os.listdir(data_dir)
for con in existing_list:
    if not os.path.exists(os.path.join(data_dir, con, "benchmark_results.pkl")):
        existing_list.remove(con)

for file in os.listdir("."):
    if file[:4] == "con_":
        if file[4:-3] in existing_list:
            print(f"Robustness benchmark data for {file} found.")
        else:
            print(f"Creating benchmarks for new controller {file}")

            controller_arg = file[:-3]
            controller_name = controller_arg[4:]
            imp = importlib.import_module(controller_arg)
            controller = imp.controller

            save_dir = os.path.join(data_dir, f"{controller_name}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            benchmark_controller(controller, save_dir)

            recompute_leaderboard = True

if recompute_leaderboard:
    src_dir = "."
    save_to = os.path.join(data_dir, "leaderboard.csv")

    leaderboard_data = []

    for f in os.listdir(src_dir):
        if f[:4] == "con_":
            mod = importlib.import_module(f[:-3])
            if hasattr(mod, "leaderboard_config"):
                conf = mod.leaderboard_config
                if os.path.exists(os.path.join(data_dir, conf["name"], "benchmark_results.pkl")):
                    print(f"Found leaderboard_config and data for {mod.leaderboard_config['name']}")

                    scores = get_scores(os.path.join(data_dir, conf["name"]), "benchmark_results.pkl")

                    final_score = (
                        0.2 * scores["model"]
                        + 0.2 * scores["measurement_noise"]
                        + 0.2 * scores["u_noise"]
                        + 0.2 * scores["u_responsiveness"]
                        + 0.2 * scores["delay"]
                    )

                    if link_base != "":
                        if "simple_name" in conf.keys():
                            name_with_link = (
                                f"[{conf['simple_name']}]({link_base}{conf['name']}/README.md)"
                            )
                        else:
                            name_with_link = f"[{conf['name']}]({link_base}{conf['name']}/README.md)"
                    else:
                        if "simple_name" in conf.keys():
                            name_with_link = conf["simple_name"]
                        else:
                            name_with_link = conf["name"]

                    append_data = [
                        name_with_link,
                        conf["short_description"],
                        "{:.1f}".format(100*scores["model"]),
                        "{:.1f}".format(100*scores["measurement_noise"]),
                        "{:.1f}".format(100*scores["u_noise"]),
                        "{:.1f}".format(100*scores["u_responsiveness"]),
                        "{:.1f}".format(100*scores["delay"]),
                        "{:.3f}".format(final_score),
                        conf["username"],
                    ]
                    if link_base != "":
                        append_data.append("[Data and Plots](" + link_base + conf["name"] + ")")

                    leaderboard_data.append(append_data)

    header = "Controller,Short Controller Description,Model [%],Velocity Noise [%], Torque Noise [%], Torque Step Response [%], Time delay [%],Overall Robustness Score,Username"

    if link_base != "":
        header += ",Data"

    np.savetxt(
        save_to,
        leaderboard_data,
        header=header,
        delimiter=",",
        fmt="%s",
        comments="",
    )

    print(pandas.read_csv(save_to).sort_values(by=["Overall Robustness Score"], ascending=False).to_markdown(index=False))
