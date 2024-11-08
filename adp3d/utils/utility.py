class DotDict(dict):
    """
    A dictionary subclass that supports attribute-style access.

    This class allows you to access dictionary keys as if they were attributes.
    For example, instead of writing `d['key']`, you can write `d.key`.

    Example usage:
        d = DotDict()
        d.key = 'value'
        print(d.key)  # Output: value
        print(d['key'])  # Output: value

    From: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def try_gpu():
    """Attempt to select the free-est GPU (by memory) for use with PyTorch.

    Returns
    -------
    torch.device
        GPU with most available memory, or the CPU if no GPUs are available.
    """
    import subprocess
    from io import StringIO
    import pandas as pd
    import torch
    try:
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        gpu_stats_str = gpu_stats.decode('utf-8')
        gpu_df = pd.read_csv(StringIO(gpu_stats_str),
                             names=['memory.used', 'memory.free'],
                             skiprows=1)
        print('GPU usage:\n{}'.format(gpu_df))
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
        if gpu_df.empty:
            print("No GPUs found.")
            return torch.device('cpu')
        idx = gpu_df['memory.free'].idxmax()
        print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
        
        return torch.device('cuda:{}'.format(idx))
    
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Failed to run nvidia-smi: {e}")
        return None

    