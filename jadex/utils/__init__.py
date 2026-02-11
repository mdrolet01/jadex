import numpy as np
from flax import struct
from omegaconf import DictConfig


def non_pytree(*args, **kwargs):
    return struct.field(*args, pytree_node=False, **kwargs)


def is_power_of_two(n):
    power_of_2 = bool(n > 0 and (n & (n - 1)) == 0)
    return power_of_2


def get_closest_square(full_size):
    target = int(np.sqrt(full_size))

    def closest_square(size, dir="split"):
        if size < 1 or size >= full_size:
            return np.inf
        if full_size % size == 0:
            return size
        else:
            if dir == "split":
                closest_down = closest_square(size - 1, "down")
                closest_up = closest_square(size + 1, "up")
                if np.abs(closest_down - target) < np.abs(closest_up - target):
                    return closest_down
                else:
                    return closest_up
            elif dir == "down":
                return closest_square(size - 1, "down")
            elif dir == "up":
                return closest_square(size + 1, "up")

    dim1 = closest_square(target)
    return (dim1, full_size // dim1)


def mplfig_to_npimage(fig):
    """Converts a matplotlib figure to a RGB frame after updating the canvas"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # update/draw the elements

    # get the width and the height to resize the matrix
    l, b, w, h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    # Get the RGBA buffer and convert to RGB
    buf = np.asarray(canvas.buffer_rgba())
    # Remove alpha channel
    image = buf[:, :, :3]
    return image


def submit_job(fn, cfg: DictConfig):
    cluster_cfg = cfg.cluster

    if cfg.cluster.id != "local":
        import submitit

        executor = submitit.SlurmExecutor(folder="slurm_out/%j")
        cluster_cfg_dict = dict(cluster_cfg)
        cluster_cfg_dict.pop("submitted", None)
        cluster_cfg_dict.pop("name", None)
        cluster_cfg_dict.pop("id", None)
        executor.update_parameters(**cluster_cfg_dict)
        executor.submit(fn, cfg)
        print(f"Submitted job for {cfg.job.uuid}")
    else:
        fn(cfg)
