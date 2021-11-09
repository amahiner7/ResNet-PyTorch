RESNET50_MODEL_PARAMS = [
    {"out_channels": [64, 64, 256],
     "kernel_size": [(1, 1), (3, 3), (1, 1)],
     "stride": [[(1, 1), (1, 1)], (1, 1), (1, 1)],
     "padding": ['valid', 'same', 'valid'],
     "num_blocks": 3
     },
    {"out_channels": [128, 128, 512],
     "kernel_size": [(1, 1), (3, 3), (1, 1)],
     "stride": [[(2, 2), (1, 1)], (1, 1), (1, 1)],
     "padding": ['valid', 'same', 'valid'],
     "num_blocks": 4
     },
    {"out_channels": [256, 256, 1024],
     "kernel_size": [(1, 1), (3, 3), (1, 1)],
     "stride": [[(2, 2), (1, 1)], (1, 1), (1, 1)],
     "padding": ['valid', 'same', 'valid'],
     "num_blocks": 6
     },
    {"out_channels": [512, 512, 2048],
     "kernel_size": [(1, 1), (3, 3), (1, 1)],
     "stride": [[(2, 2), (1, 1)], (1, 1), (1, 1)],
     "padding": ['valid', 'same', 'valid'],
     "num_blocks": 3
     }
]
