"""Ref: https://github.com/sniklaus/pytorch-hed"""

import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Processor(nn.Module):
    def __init__(self):
        super(Processor, self).__init__()

    def forward(img):
        img = np.float32(img) / 255.0
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 4:
            img = np.transpose(img, (0, 3, 1, 2))
        else:
            raise ValueError("The dim of inputs must be 3 or 4")
        img = np.ascontiguousarray(img)
        return img

    def inverse(img):
        img = np.squeeze(np.float32(img), axis=0) \
            if img.shape[0] == 1 else img
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 4:
            img = np.transpose(img, (0, 2, 3, 1))
        img = np.clip(img * 255.0, 0, 255.0).astype('uint8')
        img = np.ascontiguousarray(img)
        return img


class HEDNet(nn.Module):
	def __init__(self, device='cuda:0'):
		super(HEDNet, self).__init__()

		self.device = device

		self.netVggOne = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netVggTwo = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netVggThr = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netVggFou = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netVggFiv = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.netScoreOne = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreTwo = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreThr = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFou = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFiv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

		self.netCombine = nn.Sequential(
			nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

		# self.checkpoint = osp.join(osp.dirname(osp.abspath(__file__)), "network-bsds500.pytorch")
		# self.load_state_dict(torch.load(self.checkpoint))
		self.model_url = 'http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch'
		state_dict = torch.hub.load_state_dict_from_url(self.model_url)
		self.load_state_dict({ k.replace('module', 'net'): w for k, w in state_dict.items() })
		self.to(self.device)

	def forward(self, img):
		b = (img[:, 0:1, :, :] * 255.0) - 104.00698793
		g = (img[:, 1:2, :, :] * 255.0) - 116.66876762
		r = (img[:, 2:3, :, :] * 255.0) - 122.67891434

		img = torch.cat([ b, g, r ], 1)

		fs1 = self.netVggOne(img)
		fs2 = self.netVggTwo(fs1)
		fs3 = self.netVggThr(fs2)
		fs4 = self.netVggFou(fs3)
		fs5 = self.netVggFiv(fs4)

		s1 = self.netScoreOne(fs1)
		s2 = self.netScoreTwo(fs2)
		s3 = self.netScoreThr(fs3)
		s4 = self.netScoreFou(fs4)
		s5 = self.netScoreFiv(fs5)

		s1 = F.interpolate(input=s1, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
		s2 = F.interpolate(input=s2, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
		s3 = F.interpolate(input=s3, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
		s4 = F.interpolate(input=s4, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
		s5 = F.interpolate(input=s5, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)

		return self.netCombine(torch.cat([ s1, s2, s3, s4, s5 ], 1)), ( s1, s2, s3, s4, s5 )

	def predict(self, input_img_test):
		input_img_test  = Processor.forward(input_img_test)
		input_img_test  = torch.FloatTensor(input_img_test).to(self.device)
		output_img_test, *_ = self.forward(input_img_test)
		output_img_test = output_img_test.detach().cpu().numpy()
		output_img_test = Processor.inverse(output_img_test)
		return output_img_test


if __name__ == '__main__':
	from PIL import Image

	model = HEDNet(device=torch.device('cuda:0')).eval()

	input_img_test = Image.open("../Madison.png")
	gray_img_test = np.uint8(input_img_test.convert('L'))
	output_img_test = model.predict(input_img_test)
	output_img_test = np.concatenate([gray_img_test, output_img_test.squeeze(2)], axis=1)
	Image.fromarray(output_img_test).save("demo.png")
