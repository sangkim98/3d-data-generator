REMOVE = 'REMOVE'

MDM_JOINT_MAP = {
    'Hips': 0, 'LeftUpLeg': 1, 'RightUpLeg': 2, 'Spine': 3, 'LeftLeg': 4, 'RightLeg': 5,
    'Spine1': 6, 'LeftFoot': 7, 'RightFoot': 8, 'Spine2': 9, 'LeftToeBase': 10, 'RightToeBase': 11,
    'Neck': 12, 'LeftShoulder': 13, 'RightShoulder': 14, 'Head': 15, 'LeftArm': 16, 'RightArm': 17,
    'LeftForeArm': 18, 'RightForeArm': 19, 'LeftHand': 20, 'RightHand': 21,
}

MDM_JOINT_PAIRS = [
    ['Hips', 'Spine'], ['Spine', 'Spine1'], ['Spine1', 'Spine2'],
    ['Spine2', 'Neck'], ['Neck', 'Head'], ['Spine2', 'RightShoulder'],
    ['RightShoulder', 'RightArm'], ['RightArm','RightForeArm'], ['RightForeArm', 'RightHand'],
    ['Hips', 'RightUpLeg'], ['RightUpLeg', 'RightLeg'], ['RightLeg', 'RightFoot'],
    ['RightFoot', 'RightToeBase'], ['Spine2', 'LeftShoulder'], ['LeftShoulder', 'LeftArm'],
    ['LeftArm', 'LeftForeArm'], ['LeftForeArm', 'LeftHand'], ['Hips', 'LeftUpLeg'],
    ['LeftUpLeg', 'LeftLeg'], ['LeftLeg', 'LeftFoot'], ['LeftFoot', 'LeftToeBase']
]

OPENPOSE_JOINT_MAP = {
    'Nose': 0, 'Neck': 1,
    'RShoulder': 2, 'RElbow': 3, 'RWrist': 4,
    'LShoulder': 5, 'LElbow': 6, 'LWrist': 7,
    'RHip': 8, 'RKnee': 9, 'RAnkle': 10,
    'LHip': 11, 'LKnee': 12, 'LAnkle': 13,
    'REye': 14, 'LEye': 15, 'REar': 16, 'LEar': 17
}

OPENPOSE_JOINT_COLOR = {
    'Nose': [255, 0, 0], 'Neck': [255, 85, 0],
    'RShoulder': [255, 170, 0], 'RElbow': [255, 255, 0], 'RWrist': [170, 255, 0],
    'LShoulder': [85, 255, 0], 'LElbow': [0, 255, 0], 'LWrist': [0, 255, 85],
    'RHip': [0, 255, 170], 'RKnee': [0, 255, 255], 'RAnkle': [0, 170, 255],
    'LHip': [0, 85, 255], 'LKnee': [0, 0, 255], 'LAnkle': [85, 0, 255],
    'REye': [170, 0, 255], 'LEye': [255, 0, 255], 'REar': [255, 0, 170], 'LEar': [255, 0, 85]
}

OPENPOSE_JOINT_PAIRS = [
    ['Neck', 'RShoulder'], ['Neck', 'LShoulder'], ['RShoulder', 'RElbow'],
    ['RElbow', 'RWrist'], ['LShoulder', 'LElbow'], ['LElbow', 'LWrist'],
    ['Neck', 'RHip'], ['RHip', 'RKnee'], ['RKnee', 'RAnkle'],
    ['Neck', 'LHip'], ['LHip', 'LKnee'], ['LKnee', 'LAnkle'],
    ['Neck', 'Nose'], ['Nose', 'REye'], ['REye', 'REar'],
    ['Nose', 'LEye'], ['LEye', 'LEar']
]

OPENPOSE_JOINT_PAIRS_COLOR = [
    [153, 0, 0], [153, 51, 0], [153, 102, 0],
    [153, 153, 0], [102, 153, 0], [51, 153, 0],
    [0, 153, 0], [0, 153, 51], [0, 153, 102],
    [0, 153, 153], [0, 102, 153], [0, 51, 153],
    [0, 0, 153], [51, 0, 153], [102, 0, 153],
    [153, 0, 153], [153, 0, 102]
]

MDM2OPENPOSE_KEYVAL = {
    'Hips': REMOVE, 'LeftUpLeg': REMOVE, 'RightUpLeg': REMOVE, 'Spine': REMOVE,
    'LeftLeg': 'LKnee', 'RightLeg': 'RKnee', 'Spine1': REMOVE, 'LeftFoot': 'LAnkle',
    'RightFoot': 'RAnkle', 'Spine2': REMOVE, 'LeftToeBase': REMOVE, 'RightToeBase': REMOVE,
    'Neck': REMOVE, 'LeftShoulder': REMOVE, 'RightShoulder': REMOVE, 'Head': 'Nose',
    'LeftArm': 'LShoulder', 'RightArm': 'RShoulder', 'LeftForeArm': 'LElbow', 'RightForeArm': 'RElbow',
    'LeftHand': 'LWrist', 'RightHand': 'RWrist'
}