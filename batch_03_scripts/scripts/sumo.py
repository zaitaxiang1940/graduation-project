import os
import sys
from _settings import Settings
import logging
import shutil
from trajectory.utils.paths import build_project_path


def _ensure_sumo_tools_on_path():
    sumo_home = os.environ.get("SUMO_HOME")
    candidates = [
        sumo_home,
        "/usr/share/sumo",
        "/usr/local/share/sumo",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        tools_dir = os.path.join(candidate, "tools")
        if os.path.isdir(tools_dir):
            os.environ["SUMO_HOME"] = candidate
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            return candidate
    raise RuntimeError(
        "SUMO 未安装或 SUMO_HOME 未正确设置。"
        "Ubuntu 24.04 可执行: sudo apt install -y sumo sumo-tools sumo-doc "
        "并设置: export SUMO_HOME=/usr/share/sumo"
    )

# Must be done after SUMO_HOME is set
try:
    _ensure_sumo_tools_on_path()
    import traci
    _TRACI_IMPORT_ERROR = None
except Exception as e:
    traci = None
    _TRACI_IMPORT_ERROR = e


def get_sumo_binary():
    sumo_binary = "sumo"
    if Settings.USE_GUI:
        if Settings.SYSTEM == "Windows":
            sumo_binary = "sumo-gui.exe"
        elif Settings.SYSTEM == "Linux":
            sumo_binary = "sumo-gui"
    else:
        if Settings.SYSTEM == "Windows":
            sumo_binary = "sumo.exe"
        elif Settings.SYSTEM == "Linux":
            sumo_binary = "sumo"
    return sumo_binary


def start_sumo():
    if traci is None:
        raise RuntimeError(str(_TRACI_IMPORT_ERROR))
    sumo_binary = get_sumo_binary()
    sumo_binary = shutil.which(sumo_binary) or sumo_binary
    sumo_cmd = [sumo_binary, "-c", str(build_project_path("scripts", "ramp.sumocfg")), "--step-length", str(Settings.TICK_LENGTH)]
    if Settings.USE_ALTERNATE_TRAFFIC_DISTRIBUTION:
        if Settings.TRAFFIC_DENSITY == "low":
            sumo_cmd.extend(["--route-files", str(build_project_path("scripts", "merge2.rou.xml"))])
        elif Settings.TRAFFIC_DENSITY == "medium":
            sumo_cmd.extend(["--route-files", str(build_project_path("scripts", "merge2b.rou.xml"))])
        elif Settings.TRAFFIC_DENSITY == "high":
            sumo_cmd.extend(["--route-files", str(build_project_path("scripts", "merge2c.rou.xml"))])
        else:
            raise ValueError("Unknown TRAFFIC_DENSITY: {}".format(Settings.TRAFFIC_DENSITY))
    elif Settings.USE_SIMPLE_TRAFFIC_DISTRIBUTION:
        sumo_cmd.extend(["--route-files", str(build_project_path("scripts", "merge_impossible.rou.xml"))])
    if Settings.SEED != "Random":
        sumo_cmd.extend(["--seed", str(Settings.SEED)])
    else:
        sumo_cmd.extend(["--random"])
    try:
        port = int(os.environ.get("TRACI_PORT", "8813"))
        traci.start(sumo_cmd, port=port)
    except FileNotFoundError as e:
        import traceback
        print(traceback.format_exc())
        print("Have you installed SUMO and added the directory containing sumo or sumo.exe to your PATH?")
    if Settings.USE_SIMPLE_TRAFFIC_DISTRIBUTION:
        traci.vehicletype.setMaxSpeed("normal", Settings.OTHER_CAR_SPEED)


def exit_sumo():
    traci.close()


def change_step_size(new_step_size):
    exit_sumo()
    Settings.TICK_LENGTH = new_step_size
    start_sumo()
