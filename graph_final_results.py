import json
import numpy as np
import matplotlib.pyplot as plt

paths = {"fixed": {"act": {"both": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/pretrain_both_20",
                                    "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/no_pretrain_both_20"},
                           "vision": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/pretrain_vision_only_20",
                                      "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/no_pretrain_vision_only_20"},
                           "tactile": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/pretrain_gel_only_20",
                                       "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/ACT/no_pretrain_gel_only_20"}},
                   "diffusion": {"both": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/pretrain_both_20",
                                          "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/no_pretrain_both_20"},
                                 "vision": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/pretrain_vision_only_20",
                                            "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/no_pretrain_vision_only_20"}}}}
                                #  "tactile": {"pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/pretrain_gel_only_20",
                                #              "non-pretrained": "/home/aigeorge/research/TactileACT/data/Final Trained Policies/Fixed/Diffusion/no_pretrain_gel_only_20"}}}}

success_rates = {"fixed": {"act": {"both": {"pretrained": 0.95, "non-pretrained": 0.9},
                                  "vision": {"pretrained": 0.85, "non-pretrained": 0.20},
                                  "tactile": {"pretrained": 0.45, "non-pretrained": 0.70}},
                           "diffusion": {"both": {"pretrained": 0.75, "non-pretrained": 0.70},
                                         "vision": {"pretrained": 0.75, "non-pretrained": 0.45}}}}


def print_final_results():
    # print the results of the final trained policies for the fixed dataset using the ACT model:
    num_runs = 20
    for camera_type in paths["fixed"]["act"]:
        for pretrain in paths["fixed"]["act"][camera_type]:
            success = 0
            for i in range(1, 1+num_runs):
                run_stats_file = f"{paths['fixed']['act'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
                with open(run_stats_file, 'r') as f:
                    run_stats = json.load(f)
                if run_stats["was_successful"] == "y":
                    success += 1
                elif run_stats["was_successful"] == "n":
                    pass
                else:
                    raise ValueError(f"Unexpected value for was_successful: {run_stats['was_successful']}")
            print(f"ACT: Camera type: {camera_type}, Pretrain: {pretrain}, Success rate: {success}/{num_runs} ({100*success/num_runs:.2f}%)")

    for camera_type in paths["fixed"]["diffusion"]:
        for pretrain in paths["fixed"]["diffusion"][camera_type]:
            success = 0
            for i in range(1, 1+num_runs):
                run_stats_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
                with open(run_stats_file, 'r') as f:
                    run_stats = json.load(f)
                if run_stats["was_successful"] == "y":
                    success += 1
                elif run_stats["was_successful"] == "n":
                    pass
                else:
                    raise ValueError(f"Unexpected value for was_successful: {run_stats['was_successful']}")
            print(f"DIffusion: Camera type: {camera_type}, Pretrain: {pretrain}, Success rate: {success}/{num_runs} ({100*success/num_runs:.2f}%)")
            


def plot_percentiles():
    plt.figure()
    plt.title("Percentile Average Stress")
    num_runs = 20
    for camera_type in paths["fixed"]["diffusion"]:
        for pretrain in paths["fixed"]["diffusion"][camera_type]:
            mean_strains = []
            for i in range(1, 1+num_runs):
                run_stats_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
                gelsight_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/gelsight_strains.npy"
                with open(run_stats_file, 'r') as f:
                    run_stats = json.load(f)
                strains = np.load(gelsight_file)
                closed_gripper_strains = strains[np.where(strains[:, 0] > 0.7)]
                # plt.plot(closed_gripper_strains[:, 1], label=f"1")
                # plt.plot(closed_gripper_strains[:, 2], label=f"2")
                # plt.legend()
                # plt.show()
                # mean_strains.append(np.mean(strains, axis=1))
                mean_strains.append(closed_gripper_strains[:, 2])
                # mean_strains = strains[:, 1]
            
            mean_strains = np.concatenate(mean_strains, axis=0)
            print('median for ', camera_type, pretrain, np.median(mean_strains))
            percentiles = np.percentile(mean_strains, range(101))
            plt.plot(percentiles, label=f"{camera_type}, {pretrain}")

    plt.legend()
    plt.show()

def plot_percentiles_diff_avg():
    plt.figure()
    plt.title("Percentile Average Stress")
    num_runs = 20
    for camera_type in paths["fixed"]["diffusion"]:
        for pretrain in paths["fixed"]["diffusion"][camera_type]:
            mean_strains = []
            for i in range(1, 1+num_runs):
                run_stats_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
                gelsight_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/gelsight_strains.npy"
                with open(run_stats_file, 'r') as f:
                    run_stats = json.load(f)
                strains = np.load(gelsight_file)
                closed_gripper_strains = strains[np.where(strains[:, 0] > 0.7)]
                # plt.plot(closed_gripper_strains[:, 1], label=f"1")
                # plt.plot(closed_gripper_strains[:, 2], label=f"2")
                # plt.legend()
                # plt.show()
                # mean_strains.append(np.mean(strains, axis=1))
                mean_strains.append(closed_gripper_strains[:, 2])
                # mean_strains = strains[:, 1]
            
            mean_strains = np.concatenate(mean_strains, axis=0)
            percentiles = np.percentile(mean_strains, range(101))
            print('median for ', camera_type, pretrain, np.median(mean_strains))
            plt.plot(percentiles, label=f"{camera_type}, {pretrain}")

    plt.legend()
    plt.show()

def plot_percentiles_diffusion():
    num_runs = 20
    plt.figure()
    plt.title("Percentile Max Stress")
    for camera_type in paths["fixed"]["diffusion"]:
        for pretrain in paths["fixed"]["diffusion"][camera_type]:
            max_strains = []
            for i in range(1, 1+num_runs):
                run_stats_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/run_stats.json"
                gelsight_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/gelsight_max_strains.npy"
                gelsight_avg_file = f"{paths['fixed']['diffusion'][camera_type][pretrain]}/run_data/run_{i}/gelsight_strains.npy"
                with open(run_stats_file, 'r') as f:
                    run_stats = json.load(f)
                strains = np.load(gelsight_file)
                avg_strains = np.load(gelsight_avg_file)
                max_strains.append(strains)
            
            max_strains = np.concatenate(max_strains, axis=0)
            percentiles = np.percentile(max_strains, range(101))
            plt.plot(percentiles, label=f"{camera_type}, {pretrain}")
    
    plt.legend()
    plt.show()

# make a bar chart of the success rate, with a blue bar for ACT and a green bar for Diffusion. 
# The bars should be grouped by camera type, with the pre-trained and non-pretrained models side by side.
# Pretrained models should be darker than non-pretrained models.
def plot_results():
    fig, ax = plt.subplots()
    width = 0.2
    x = [0, 1, 2]

    act_color = "lightblue"
    act_edge = "black"
    diff_color = "lightgreen"
    diff_edge = "black"
    pretrained_hatch = "//"
    non_pretrained_hatch = ""

    # Tactile + Vision
    act_both = [success_rates["fixed"]["act"]["both"]["pretrained"], success_rates["fixed"]["act"]["both"]["non-pretrained"]]
    diff_both = [success_rates["fixed"]["diffusion"]["both"]["pretrained"], success_rates["fixed"]["diffusion"]["both"]["non-pretrained"]]
    ax.bar(x[0] - 1.6*width, act_both[1]*100, width, color=act_color, hatch=non_pretrained_hatch, edgecolor=act_edge)
    ax.bar(x[0] - 0.6*width, act_both[0]*100, width, color=act_color, hatch=pretrained_hatch, edgecolor=act_edge)
    ax.bar(x[0] + 0.6*width, diff_both[1]*100, width, color=diff_color, hatch=non_pretrained_hatch, edgecolor=diff_edge)
    ax.bar(x[0] + 1.6*width, diff_both[0]*100, width, color=diff_color, hatch=pretrained_hatch, edgecolor=diff_edge)
    

    ax.legend(["ACT (not pretrained)", "ACT (pretrained)", "Diffusion (not pretrained)", "Diffusion (pretrained)"])

    act_both = [success_rates["fixed"]["act"]["vision"]["pretrained"], success_rates["fixed"]["act"]["vision"]["non-pretrained"]]
    diff_both = [success_rates["fixed"]["diffusion"]["vision"]["pretrained"], success_rates["fixed"]["diffusion"]["vision"]["non-pretrained"]]
    ax.bar(x[1] - 1.6*width, act_both[1]*100, width, color=act_color, hatch=non_pretrained_hatch, edgecolor='black')
    ax.bar(x[1] - 0.6*width, act_both[0]*100, width, color=act_color, hatch=pretrained_hatch, edgecolor='black')
    ax.bar(x[1] + 0.6*width, diff_both[1]*100, width, color=diff_color, hatch=non_pretrained_hatch, edgecolor='black')
    ax.bar(x[1] + 1.6*width, diff_both[0]*100, width, color=diff_color, hatch=pretrained_hatch, edgecolor='black')
    
    act_both = [success_rates["fixed"]["act"]["tactile"]["pretrained"], success_rates["fixed"]["act"]["tactile"]["non-pretrained"]]
    ax.bar(x[2] - 1.5*width, act_both[1]*100, width, color=act_color, hatch=non_pretrained_hatch, edgecolor=act_edge)
    ax.bar(x[2] - 0.5*width, act_both[0]*100, width, color=act_color, hatch=pretrained_hatch, edgecolor=act_edge)
    

    # add x axis labels
    ax.set_xticks([x[0], x[1], x[2]-width])
    ax.set_xticklabels(["Tactile + Vision", "Vision Only", "Tactile Only"])

    # set y axis limits (so that the legend doesn't overlap with the bars)
    ax.set_ylim(0, 117)

    # set the y ticks to be in increments of 0.1 (max at 1)
    ax.set_yticks(np.arange(0, 120, 20))

    # set the y axis label
    ax.set_ylabel("Success Rate (%)")

    plt.show()


if __name__ == "__main__":
    print_final_results()
    # plot_results()
    # plot_percentiles_diffusion()
    plot_percentiles()
    plot_percentiles_diff_avg()


