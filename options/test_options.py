from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # Testing specific options
        parser.add_argument('--model_path', type=str, required=True,
                            help='Path to trained MHA model checkpoint')
        parser.add_argument('--no_resize', action='store_true',
                            help='Do not resize images during testing')
        parser.add_argument('--no_crop', action='store_true',
                            help='Do not crop images during testing')
        parser.add_argument('--eval', action='store_true',
                            help='use eval mode during test time')

        # Single image testing
        parser.add_argument('--image_path', type=str, default=None,
                            help='Path to single image for testing (leave empty for dataset mode)')

        # Batch testing
        parser.add_argument('--test_splits', type=str, nargs='+',
                            default=['val'],
                            help='List of test dataset splits (e.g., val test)')
        parser.add_argument('--results_dir', type=str, default='./results',
                            help='Directory to save test results')

        # Wavelet parameters
        parser.add_argument('--wavelet_type', type=str, default='haar',
                            help='Wavelet type (default: haar)')
        parser.add_argument('--wavelet_level', type=int, default=3,
                            help='Wavelet decomposition level (default: 3)')

        self.isTrain = False
        return parser
