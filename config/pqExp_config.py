import numpy as np
pqExp_config = {
    'Brain': {
        'neurons_per_layer': np.array([16, 32, 32]),
        'learning_rate': 0.00025,
        'output_graph': False,
        'restore': False
    },
    'Agent': {
        'reward_decay': 0.95,
        'replace_target_iter': 1000,
        'memory_size': 100000,
        'batch_size': 32,
        'MAX_EPSILON': 0.9,
        'MIN_EPSILON': 0.1,
        'LAMBDA': 0.0001,

    }
}
