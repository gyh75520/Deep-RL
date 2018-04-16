import numpy as np
configs = {
    'Brain': {
        'filters_per_layer': np.array([32, 64, 64]),
        'kernel_size_per_layer': [(8, 8), (4, 4), (3, 3)],
        'conv_strides_per_layer': [(4, 4), (2, 2), (1, 1)],
        'learning_rate': 0.00025,
        'output_graph': False,
        'restore': False
    },
    'Agent': {
        'reward_decay': 0.99,
        'replace_target_iter': 10000,
        'memory_size': 1000000,
        'batch_size': 32,
        'MAX_EPSILON': 1,
        'MIN_EPSILON': 0.1,
        'LAMBDA': 0.0000002,
    },
    'ExperienceReplay': {
        'replay_start_size': 50000,
        'update_frequency': 4,
    }
}
