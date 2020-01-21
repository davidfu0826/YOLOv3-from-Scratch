from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import numpy as numpy

from util import *

def get_test_input():
	img = cv2.imread("dog-cycle-car.png")
	img = cv2.resize(img, (416,416)) 			# Resize to the input dimension
	img_ = img[:,:,::-1].transpose((2,0,1)) 	# BGR -> RGB | H x W x C -> C x H x W
	img_ = img_[np.newaxis,:,:,:]/255.0			# Add a channel at 0 (for batch) | Normalise
	img_ = torch.from_numpy(img_).float()		# Convert to float
	img_ = Variable(img_)
	return img_

def parse_cfg(cfgfile):
	"""
	Takes a configuration file

	Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
	
	"""

	file = open(cfgfile, 'r')
	lines = file.read().split('\n')              # store the lines in a list
	lines = [x for x in lines if len(x) > 0]     # get rid of the empty lines
	lines = [x for x in lines if x[0] != '#']    # remove comments
	lines = [x.rstrip().lstrip() for x in lines] # remove blankspaces on the sides

	block = dict()
	blocks = list()

	for line in lines:
		if line[0] == "[":
			if len(block) != 0:
				blocks.append(block)
				block = dict()
			block['type'] = line[1:-1].rstrip()
		else:
			key, value = line.split("=")
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)

	return blocks

def create_modules(blocks):
	net_info = blocks[0]
	module_list = nn.ModuleList()
	prev_filters = 3 # The input image is RGB-valued
	output_filters = []

	for index, x in enumerate(blocks[1:]):
		module = nn.Sequential()

		#check type of block
		#create a new module for the block
		#append to module_list

		if (x["type"] == "convolutional"):
			activation = x['activation']
			try:
				batch_normalize = int(x['batch_normalize'])
				bias = False
			except:
				batch_normalize = 0
				bias = True

			filters = int(x["filters"])
			padding = int(x["pad"])
			kernel_size = int(x["size"])
			stride = int(x["stride"])

			if padding:
				pad = (kernel_size - 1) // 2
			else:
				pad = 0

			# Add the convolutional layer
			conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
			module.add_module(f"conv_{index}", conv)

			# Add the Batch Norm Layer
			if batch_normalize:
				bn = nn.BatchNorm2d(filters)
				module.add_module(f"batch_norm_{index}", bn)

			# Check the activation.
			# It is either linear or Leaky ReLU for YOLO
			if activation == "leaky":
				activn = nn.LeakyReLU(0.1, inplace = True)
				module.add_module(f"leaky_{index}", activn)

		# If it's an upsampling layer
		# We use Bilinear2dUpsampling
		elif (x["type"] == "upsample"):
			stride = int(x["stride"])
			upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
			module.add_module(f"upsample_{index}", upsample)

		# If it is a route layer
		elif (x["type"] == "route"):
			x["layers"] = x["layers"].split(',')
			# Start of a route
			start = int(x["layers"][0])
			# end, if there exist one
			try:
				end = int(x["layers"][1])
			except:
				end = 0
			# Positive annotation
			if start > 0:
				start = start - index
			if end > 0:
				end = end - index
			route = EmptyLayer()
			module.add_module(f"route_{index}", route)
			if end < 0:
				# If concatenating maps
				filters = output_filters[index + start] + output_filters[index + end]
			else:
				filters = output_filters[index + start]

		# Shortcut corresponds to skip connection
		elif x["type"] == "shortcut":
			shortcut = EmptyLayer()
			module.add_module(f"shortcut_{index}", shortcut)

		# YOLO is the detection layer
		elif x["type"] == "yolo":
			mask = x["mask"].split(",")
			mask = [int(x) for x in mask]

			anchors = x["anchors"].split(",")
			anchors = [int(a) for a in anchors]
			anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in mask]

			detection = DetectionLayer(anchors)
			module.add_module(f"Detection_{index}", detection)

		module_list.append(module)
		prev_filters = filters
		output_filters.append(filters)

	return (net_info, module_list)

class EmptyLayer(nn.Module):
	def __init__(self):
		super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors

class Darknet(nn.Module):
	def __init__(self, cfgfile):
		super(Darknet, self).__init__()
		self.blocks = parse_cfg(cfgfile)
		self.net_info, self.module_list = create_modules(self.blocks)

	def forward(self, x, CUDA):
		"""
		Takes an input x and if we use GPU, we pass True.

		Returns the forward pass output.

		"""
		modules = self.blocks[1:]
		outputs = {} # We cache the outputs for the route layer

		write = 0 # This is explained a bit later
		for i, module in enumerate(modules): # The order is kept in module_list so we can iterate over modules
			module_type = (module["type"])

			if module_type == "convolutional" or module_type == "upsample":
				x = self.module_list[i](x)
				
			elif module_type == "route":
				layers = module["layers"]	
				layers = [int(layer) for layer in layers]

				if (layers[0]) > 0:
					layers[0] = layers[0] - i

				if len(layers) == 1:
					x = output[i + (layers[0])]

				else:
					if (layers[1]) > 0:
						layers[1] = layers[1] - i

					map1 = outputs[i + layers[0]]
					map2 = outputs[i + layers[1]]

					x = torch.cat((map1, map2), 1)

			elif module_type == "shortcut":
				from_ = int(module["from"])
				x = outputs[i-1] + outputs[i+from_]

			elif module_type == "yolo":

				anchors = self.module_list[i][0].anchors
				# Get the input dimensions
				inp_dim = int (self.net_info["height"])

				# Get the number of classes
				num_classes = int (module["classes"])

				# Transform
				x = x.data
				x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
				if not write:
					detections = x
					write = 1

				else:
					detections = torch.cat((detections, x), 1)

			outputs[i] = x
		return detections





if __name__ == '__main__':
	model = Darknet("cfg/yolov3.cfg")
	inp = get_test_input()
	pred = model(inp, torch.cuda.is_available())
	print(pred)


