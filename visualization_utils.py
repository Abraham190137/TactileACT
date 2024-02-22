import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class DebugController:
    def __init__(self, print=False, plot=False, epoch=0, batch=0, dataset='train', visualizations_dir = None):
        self.print = print
        self.plot = plot
        self.epoch = epoch
        self.batch = batch
        self.dataset = dataset
        self._visualizations_dir = visualizations_dir
        self.std = None
        self.mean = None

    @property
    def visualizations_dir(self):
        if self._visualizations_dir is None:
            self._visualizations_dir = self.gen_visualizations_dir()
            return self._visualizations_dir

        else:
            return self._visualizations_dir
        
    @visualizations_dir.setter
    def visualizations_dir(self, value):
        if not os.path.exists(value):
            os.makedirs(value)
        self._visualizations_dir = value

    def gen_visualizations_dir(self) -> str:
        n = 0
        while os.path.exists(f'visualizations/{n}'):
            n += 1
        os.makedirs(f'visualizations/{n}')
        return f'visualizations/{n}'
    

debug = DebugController()

def visualise_trajectory(env_img, trajectory, title='Trajectory'):
    image_bgr = cv2.cvtColor(env_img, cv2.COLOR_RGB2BGR)

    # Draw circles for each point in the trajectory
    for point in trajectory:
        point = tuple((point * np.array(image_bgr.shape[:2])).astype(int))
        cv2.circle(image_bgr, tuple(point), radius=5, color=(0, 255, 255), thickness=1)

    # Display the image using OpenCV
    cv2.imshow('Environment with Trajectory', image_bgr)
    cv2.waitKey(1)


def z_slider(policy, post_process, qpos, image, processed_image, z_min=-3, z_max=3):
    from matplotlib.widgets import Button, Slider
    from scipy.stats import norm
    keep_going = [True]

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax = axs[0]  # Main plot
    ax2 = axs[1]  # Second plot
    ax.set_xlabel('Time [s]')
    ax2.set_xlabel('Time [s]')

    ax.imshow(image)
    ax2.imshow(image)

    # Generate percentiles for the z samples
    quantiles = np.linspace(0.05, 0.95, 20)

    # Use the percent-point function (ppf) to get the corresponding values from the standard normal distribution
    z_values = norm.ppf(quantiles, loc=0, scale=1)

    for z_value in z_values:
        # Choose a color for the trajectory based on z using the colormap
        color = plt.cm.plasma((z_value - min(z_values)) / (max(z_values) - min(z_values)))

        all_actions = policy(qpos, processed_image, z=z_value)
        trajectory = post_process(all_actions.detach().squeeze(0).cpu().numpy())
        ax2.scatter(trajectory[:, 0] * image.shape[0],
                    trajectory[:, 1] * image.shape[1],
                    facecolors='none', edgecolors=color, marker='o')

    all_actions = policy(qpos, processed_image)
    trajectory = post_process(all_actions.detach().squeeze(0).cpu().numpy())
    points = ax.scatter(trajectory[:, 0] * image.shape[0],
                        trajectory[:, 1] * image.shape[1],
                        facecolors='none', edgecolors='green', marker='o')

    ax2.scatter(trajectory[:, 0] * image.shape[0],
                trajectory[:, 1] * image.shape[1],
                facecolors='none', edgecolors='green', marker='o')

    # Adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control z.
    slider_ax = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    freq_slider = Slider(
        ax=slider_ax,
        label='z',
        valmin=z_min,
        valmax=z_max,
        valinit=0,
        orientation='vertical'
    )

    # The function to be called anytime a slider's value changes
    def update(z):
        all_actions = policy(qpos, processed_image, z=z)
        trajectory = post_process(all_actions.detach().squeeze(0).cpu().numpy())
        points.set_offsets(trajectory * image.shape[0])
        fig.canvas.draw_idle()
        print("z:", z)

    # Register the update function with each slider
    freq_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to move onto the next rollout
    resetax = fig.add_axes([0.8, 0.025, 0.125, 0.04])
    next_button = Button(resetax, 'Next Env', hovercolor='0.975')

    # Create a button to move onto the next step
    resetax2 = fig.add_axes([0.8, 0.075, 0.125, 0.04])
    continue_button = Button(resetax2, 'Continue', hovercolor='0.975')

    def next_step(event):
        plt.close(fig)

    continue_button.on_clicked(next_step)

    def next_rollout(event):
        plt.close(fig)
        keep_going[0] = False

    next_button.on_clicked(next_rollout)

    plt.show()
    return keep_going[0]

def visualize_data(image_data, qpos_data, action_data, is_pad_data, output_data=None):
    global debug
    if not debug.plot:
        return
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    images = [img.clone().detach().cpu().numpy() for img in image_data]
    qpos = qpos_data.clone().detach().cpu().numpy()
    actions = action_data.clone().detach().cpu().numpy()
    is_pad = is_pad_data.clone().detach().cpu().numpy()

    # unnormlize actions and qpos
    actions = actions * debug.std + debug.mean
    qpos = qpos * debug.std + debug.mean

    if output_data is not None:
        output = output_data.clone().detach().cpu().numpy()
        output = output * debug.std + debug.mean

    # Create a figure and axes
    fig = plt.figure(figsize=(10, 10), layout='tight')
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    axs_left = subfigs[0].subplots(len(images), 1)
    for i, image in enumerate(images):
        axs_left[i].imshow(image.transpose(1, 2, 0))     

    # Make a 3D scatter plot of the actions in the right subplot. Use cmaps to color the points based on the index
    c = np.arange(len(actions))
    ax2 = subfigs[1].add_subplot(111, projection='3d')
    # ax2.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c='b', marker='o')
    sc = ax2.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c=c, cmap='viridis', marker='o')
    if output_data is not None:
        # ax2.scatter(output[:, 0], output[:, 1], output[:, 2], c='g', marker='x')
        ax2.scatter(output[:, 0], output[:, 1], output[:, 2], c=c, cmap = 'viridis', marker='x')
    ax2.scatter(qpos[0], qpos[1], qpos[2], c='r', marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Actions and Qpos')
    cbar = fig.colorbar(sc, ax=ax2, label='Time', shrink=0.5)

    # Set the axis limits
    center = np.array([0.5, 0, 0.2])
    radius = 0.15
    ax2.set_xlim(center[0] - radius, center[0] + radius)
    ax2.set_ylim(center[1] - radius, center[1] + radius)
    ax2.set_zlim(center[2] - radius, center[2] + radius)

    # Show or save the figure
    fig.suptitle(f'Epoch {debug.epoch}')
    fig.savefig(f'{debug.visualizations_dir}/epoch_{debug.epoch} - {debug.dataset}.png')
    plt.close(fig)
    print('saved visualization to', f'{debug.visualizations_dir}/epoch_{debug.epoch} - {debug.dataset}.png')

