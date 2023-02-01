import cv2
import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F

from nets import Model

device = 'cuda'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference_init(left, right, model, n_iter=20):
	print("Model Forwarding...")
	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)

	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	# print(imgR_dw2.shape)
	with torch.inference_mode():
		pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)

	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()
	final_disp = (pred_disp * 100.0).astype("uint16")

	return final_disp

def inference_realtime(left, right, model, init_pred_flow, n_iter=20):
	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)

	with torch.inference_mode():
		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=init_pred_flow)

	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()
	final_disp = (pred_disp * 100.0).astype("uint16")

	return final_disp

def visualize(disp):
	baseline = 75.0
	focal_length = 860.0
	depth = baseline * focal_length / (disp + sys.float_info.epsilon)

	disp_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
	disp_vis = disp_vis.astype("uint8")
	disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

	cv2.namedWindow("output", cv2.WINDOW_NORMAL)
	cv2.imshow("output", disp_vis)
	cv2.waitKey(0)


if __name__ == '__main__':
	path = "D:/datasets/cmes_data/20230119/top_view/cam3/rgb/"
	file_list = os.listdir(path)

	left_img_list = [left for left in file_list if left.endswith("left.png")]
	right_img_list = [right for right in file_list if right.endswith("right.png")]

	model_path = "models/crestereo_eth3d.pth"
	model = Model(max_disp=256, mixed_precision=False, test_mode=True)
	model.load_state_dict(torch.load(model_path), strict=True)
	model.to(device)
	model.eval()

	for left_img_name, right_img_name in zip(left_img_list, right_img_list):
		left_img = cv2.imread(path + left_img_name)
		right_img = cv2.imread(path + right_img_name)

		# undistorted_left = cv2.undistort(left_img, left_intrinsic, left_distortion)
		# undistorted_right = cv2.undistort(right_img, right_intrinsic, right_distortion)
		undistorted_left = left_img
		undistorted_right = right_img

		start = time.time()
		disp = inference_init(undistorted_left, undistorted_right, model, n_iter=20)
		print(time.time() - start)

		# save_name_left = path + left_img_name.split("_")[0] + "_undistorted_left.png"
		# save_name_right = path + left_img_name.split("_")[0] + "_undistorted_right.png"
		splitted = left_img_name.split("_")
		save_name_disp = path + splitted[0] + "_" + splitted[1] + "_crestereo.png"

		# cv2.imwrite(save_name_left, undistorted_left)
		# cv2.imwrite(save_name_right, undistorted_right)
		cv2.imwrite(save_name_disp, disp)


