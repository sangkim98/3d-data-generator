SAME = True
REMOVE = None

MDM_JOINT_MAP = {
    'Hips': 0, 'LeftUpLeg': 1, 'RightUpLeg': 2, 'Spine': 3, 'LeftLeg': 4, 'RightLeg': 5,
    'Spine1': 6, 'LeftFoot': 7, 'RightFoot': 8, 'Spine2': 9, 'LeftToeBase': 10, 'RightToeBase': 11,
    'Neck': 12, 'LeftShoulder': 13, 'RightShoulder': 14, 'Head': 15, 'LeftArm': 16, 'RightArm': 17,
    'LeftForeArm': 18, 'RightForeArm': 19, 'LeftHand': 20, 'RightHand': 21,
}
MDM_JOINT_PAIRS = [
    ['Hips', 'Spine'], ['Spine', 'Spine1'], ['Spine1', 'Spine2'],
    ['Spine2', 'Neck'], ['Neck', 'Head'], ['Spine2', 'RightShoulder'],
    ['RightShoulder', 'RightArm'], ['RightArm', 'RightForeArm'], ['RightForeArm', 'RightHand'],
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
OPENPOSE_JOINT_PAIRS = [
    ['Head', 'Neck'], ['Neck', 'RShoulder'], ['RShoulder', 'RElbow'],
    ['RElbow', 'RWrist'], ['Neck', 'LShoulder'], ['LShoulder', 'LElbow'],
    ['LElbow', 'LWrist'], ['Neck', 'Chest'], ['Chest', 'RHip'], ['RHip', 'RKnee'],
    ['RKnee', 'RAnkle'], ['Chest', 'LHip'], ['LHip', 'LKnee'], ['LKnee', 'LAnkle']
]
MDM2OPENPOSE_KEYVAL = {
    'Hips': REMOVE, 'LeftUpLeg': 'LHip', 'RightUpLeg': 'RHip', 'Spine': REMOVE,
    'LeftLeg': 'LKnee', 'RightLeg': 'RKnee', 'Spine1': REMOVE, 'LeftFoot': 'LAnkle',
    'RightFoot': 'RAnkle', 'Spine2': 'Neck', 'LeftToeBase': REMOVE, 'RightToeBase': REMOVE,
    'Neck': SAME, 'LeftShoulder': REMOVE, 'RightShoulder': REMOVE, 'Head': REMOVE,
    'LeftArm': 'LShoulder', 'RightArm': 'RShoulder', 'LeftForeArm': 'LElbow', 'RightForeArm': 'RElbow',
    'LeftHand': 'LWrist', 'RightHand': 'RWrist'
}