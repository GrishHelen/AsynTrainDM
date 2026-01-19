import torch


def generate_timesteps(config, pipeline, type='random'):
    ts = pipeline.scheduler.timesteps  # [T...0]
    res_shape = (config.sample.batch_size, 4, 64, 64)

    if type == 'random':
        indices = torch.randint(low=0, high=len(ts), size=res_shape)
        tensor_t = ts[indices]
        return tensor_t
    else:
        raise NotImplementedError
