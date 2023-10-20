import torch
import smplx
import numpy as np
from convert2openpose import convert2openpose
from joint_format import SMPLX2OPENPOSE_IDX

JOINTS_IDX = [0,3,6,9,10,11,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]

class smplx2openpose(convert2openpose):
   def __init__(self, models_path: str, smplx_param_path: str, gender='male', scale_factor: float = 1.25, id: str = 'rendered') -> None:
      super().__init__(id, scale_factor)
      
      def create_smplx_model(models_path: str, smplx_param_path: str, gender: str):
         smplx_model_template = smplx.SMPLX(models_path, gender=gender)
         
         thuman_joints = np.load(smplx_param_path, allow_pickle=True)

         p = get_parameters(thuman_joints)

         smplx_model = smplx_model_template.forward(betas=p['betas'],
                                                    global_orient=p['global_orient'],
                                                    body_pose=p['body_pose'],
                                                    expression=p['expression'],
                                                    jaw_pose=p['jaw_pose'],
                                                    leye_pose=p['leye_pose'],
                                                    reye_pose=p['reye_pose'],
                                                   )

         return smplx_model, p['scale'], p['translation']

      model, scale, transl = create_smplx_model(models_path, smplx_param_path, gender)

      joints_reshaped = model.joints.detach().numpy().reshape(127,3)
      
      joints = np.delete(joints_reshaped[:60,:], JOINTS_IDX, axis=0)
      
      self.openpose18_joints = (joints[SMPLX2OPENPOSE_IDX]*scale) + transl
      
def get_parameters(thuman_joints):
   betas = smplx.utils.Tensor(torch.as_tensor(thuman_joints['betas']))
   global_orientation = smplx.utils.Tensor(torch.as_tensor(thuman_joints['global_orient']))
   pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['body_pose']))
   expression = smplx.utils.Tensor(torch.as_tensor(thuman_joints['expression']))
   jaw_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['jaw_pose']))
   leye_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['leye_pose']))
   reye_pose = smplx.utils.Tensor(torch.as_tensor(thuman_joints['reye_pose']))
   
   transl = np.asarray(thuman_joints['translation']).reshape(1,3)
   scale = thuman_joints['scale'][0]
   
   return {'betas': betas,
            'global_orient': global_orientation,
            'body_pose': pose,
            'expression': expression,
            'jaw_pose': jaw_pose,
            'leye_pose': leye_pose, 'reye_pose': reye_pose,
            'translation': transl, 'scale': scale
            }