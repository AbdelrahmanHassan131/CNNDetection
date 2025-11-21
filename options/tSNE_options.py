from .base_options import BaseOptions


class tSNE_Options(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # Dataset parameters
        parser.add_argument('--data_root', type=str, required=True,
                            help='Root directory containing ADM, DDIM, DDPM, DiffSwap, Real folders')
        parser.add_argument('--checkpoint_path', type=str, required=True,
                            help='Path to the trained model checkpoint (.pth file)')

        # Model parameters
        parser.add_argument('--embed_dim', type=int, default=128,
                            help='Embedding dimension')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads')
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate')
        parser.add_argument('--fusion_type', type=str, default='cross_attention',
                            choices=['cross_attention',
                                     'self_attention', 'concat'],
                            help='Type of fusion mechanism')
        parser.add_argument('--wavelet_level', type=int, default=3,
                            help='Wavelet decomposition level')

        # t-SNE parameters
        parser.add_argument('--perplexity', type=int, default=30,
                            help='t-SNE perplexity parameter')
        parser.add_argument('--n_iter', type=int, default=300,
                            help='Number of t-SNE iterations')
        parser.add_argument('--num_samples', type=int, default=None,
                            help='Maximum number of samples to use (None for all)')

        # Output parameters
        parser.add_argument('--output_dir', type=str, default='tsne_outputs',
                            help='Directory to save the t-SNE plots')

        self.isTrain = False
        return parser
