def get_channel_config():
    CHANNELSET = {
        "DRcalo1": {
            "NUM_POINT": 5000,
            "CHANNEL": [
                'DRcalo3dHits.amplitude_sum',
                'DRcalo3dHits.type',
                'DRcalo3dHits.time',
                'DRcalo3dHits.time_end',
                'DRcalo3dHits.position.x',
                'DRcalo3dHits.position.y',
                'DRcalo3dHits.position.z',
            ],
        },
        "DRpos1": {
            "NUM_POINT": 5000,
            "CHANNEL": [
                'DRcalo2dHits.amplitude',
                'DRcalo2dHits.type',
                'DRcalo2dHits.position.x',
                'DRcalo2dHits.position.y',
                'DRcalo2dHits.position.z',
            ],
        },
        "Reco1": {
            "NUM_POINT": 500,
            "CHANNEL": [
                'Reco3dHits_C.amplitude',
                'Reco3dHits_C.position.x',
                'Reco3dHits_C.position.y',
                'Reco3dHits_C.position.z',
                'Reco3dHits_S.amplitude',
                'Reco3dHits_S.position.x',
                'Reco3dHits_S.position.y',
                'Reco3dHits_S.position.z',
            ],
        },
        "amp": {
            "NUM_POINT": 1,
            "CHANNEL": [
                'C_amp',
                'S_amp',
            ],
        },
    }
    for pool in [1, 4, 8, 14, 28, 56]:
        CHANNELSET[f"amp{pool}"] = CHANNELSET["amp"]
        if (pool == 1): continue
        CHANNELSET[f"DRcalo{pool}"] = {
            "NUM_POINT": 5000,
            "CHANNEL": [
                f'DRcalo3dHits{pool}.amplitude_sum',
                f'DRcalo3dHits{pool}.type',
                f'DRcalo3dHits{pool}.time',
                f'DRcalo3dHits{pool}.time_end',
                f'DRcalo3dHits{pool}.position.x',
                f'DRcalo3dHits{pool}.position.y',
                f'DRcalo3dHits{pool}.position.z',
            ],
        }
        CHANNELSET[f"DRpos{pool}"] = {
            "NUM_POINT": 5000,
            "CHANNEL": [
                f'DRcalo2dHits{pool}.amplitude',
                f'DRcalo2dHits{pool}.type',
                f'DRcalo2dHits{pool}.position.x',
                f'DRcalo2dHits{pool}.position.y',
                f'DRcalo2dHits{pool}.position.z',
            ],
        }
        CHANNELSET[f"Reco{pool}"] = {
            "NUM_POINT": 500,
            "CHANNEL": [
                f'Reco3dHits{pool}_C.amplitude',
                f'Reco3dHits{pool}_C.position.x',
                f'Reco3dHits{pool}_C.position.y',
                f'Reco3dHits{pool}_C.position.z',
                f'Reco3dHits{pool}_S.amplitude',
                f'Reco3dHits{pool}_S.position.x',
                f'Reco3dHits{pool}_S.position.y',
                f'Reco3dHits{pool}_S.position.z',
            ],
        }

    CHANNELMAX = {
        'Reco3dHits_C.amplitude': 1,  # 0.025,
        'Reco3dHits_C.position.x': 1,
        'Reco3dHits_C.position.y': 1,
        'Reco3dHits_C.position.z': 1,
        'Reco3dHits_S.amplitude': 1,  # 0.003,
        'Reco3dHits_S.position.x': 1,
        'Reco3dHits_S.position.y': 1,
        'Reco3dHits_S.position.z': 1,
        'C_amp': 1,
        'S_amp': 1,
    }
    for key in CHANNELSET:
        for ch in CHANNELSET[key]['CHANNEL']:
            if not ch in CHANNELMAX:
                CHANNELMAX[ch] = 1

    SCALE = {
        "E_gen": 1,
        "E_dep": 1,
        "E_leak": 1,
        'GenParticles.momentum.p': 1,
        'GenParticles.momentum.phi': 1.,
        'GenParticles.momentum.theta': 1.,
        'Leak_sum': 1.,
        'C_amp': 0.02507,
        'S_amp': 0.00306,
    }
    return CHANNELSET, CHANNELMAX, SCALE
