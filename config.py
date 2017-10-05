import argparse
from utils import str2bool

arg_lists = []
parser = argparse.ArgumentParser(description='t-SNE Visualizer')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# mode param
mode_arg = add_argument_group('Setup')
mode_arg.add_argument('--num_samples', type=int, default=10000,
                            help='# of samples to compute embeddings on. Becomes slow if very high.')
mode_arg.add_argument('--num_dimensions', type=int, default=2,
                            help='# of tsne dimensions. Can be 2 or 3.')
mode_arg.add_argument('--shuffle', type=str2bool, default=True,
                            help='Whether to shuffle the data before embedding.')
mode_arg.add_argument('--compute_embeddings', type=str2bool, default=True,
                            help='Whether to compute embeddings. Do this once per sample size.')
mode_arg.add_argument('--with_images', type=str2bool, default=True,
                            help='Whether to overlay images on data points. Only works with 2D plots.')
mode_arg.add_argument('--random_seed', type=int, default=42,
                        help='Seed to ensure reproducibility')


# path params
misc_arg = add_argument_group('Path Params')
misc_arg.add_argument('--data_dir', type=str, default='./data',
                        help='Directory where data is stored')
misc_arg.add_argument('--plot_dir', type=str, default='./plots',
                        help='Directory where plots are saved')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed