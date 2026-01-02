from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, FeatureUpdater
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
from DBfusion import DBFusion
import numpy as np
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path= r"models/test.pth"
channel_dim = 64  
num_modals = 2  



for dataset_name in ["TNO"]: 
    print("\n"*2+"="*80)
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name)
    test_out_folder=os.path.join('test_results/',dataset_name)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    F_Encoder = nn.DataParallel(FeatureUpdater()).to(device)
    DetailFuseLayer = nn.DataParallel(DBFusion(channel_dim, num_modals, solver='anderson')).to(device)


    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    F_Encoder.load_state_dict(torch.load(ckpt_path)['F_Encoder'])
    Encoder.eval()
    Decoder.eval()
    F_Encoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()



    with torch.no_grad():
            for img_name in os.listdir(os.path.join(test_folder,"ir")):

                data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
                data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

                data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
                data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
                feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
                feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
                feature_JV_B, feature_JV_D,feature_JI_B, feature_JI_D = F_Encoder(feature_V_B, feature_V_D,feature_I_B, feature_I_D)
                feature_F_B = BaseFuseLayer(feature_JV_B + feature_JI_B)
                features = [feature_JI_D, feature_JV_D]
                feature_F_D, jacobian_loss, _ = DetailFuseLayer(features)
                data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
                data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
                fi = np.squeeze((data_Fuse * 255).cpu().numpy())
                fi = fi.astype(np.uint8)
                img_save(fi, img_name.split(sep='.')[0], test_out_folder, format='bmp')

