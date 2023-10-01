#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from utils import *
import unet3d
from config import models_genesis_config
#	显示进度条的包
from tqdm import tqdm

print("torch = {}".format(torch.__version__))

#	指定可见CPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#	加载配置文件
conf = models_genesis_config()
conf.display()

x_train = []
for i,fold in enumerate(tqdm(conf.train_fold)):
    file_name = "bat_"+str(conf.scale)+"_s_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    s = np.load(os.path.join(conf.data, file_name))
    x_train.extend(s)
x_train = np.expand_dims(np.array(x_train), axis=1)		#在1维插入一个维度，这个axis = 1是不是应该为0？

x_valid = []
for i,fold in enumerate(tqdm(conf.valid_fold)):
    file_name = "bat_"+str(conf.scale)+"_s_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    s = np.load(os.path.join(conf.data, file_name))
    x_valid.extend(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)

print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))

training_generator = generate_pair(x_train,conf.batch_size, conf)		#返回元组(x,y)
validation_generator = generate_pair(x_valid,conf.batch_size, conf)		#返回元组(x,y)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = unet3d.UNet3D()
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
model.to(device)

print("Total CUDA devices: ", torch.cuda.device_count())

summary(model, (1,conf.input_rows,conf.input_cols,conf.input_deps), batch_size=-1)	#batch_size=-1是啥意思
criterion = nn.MSELoss()

if conf.optimizer == "sgd":
	optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "adam":
	optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
	raise

# torch.optim.lr_scheduler.StepLR 的使用方法非常简单，我们只需要在创建优化器（optimizer）时将它作为参数传入，并指定降低学习率的策略。具体来说，StepLR 的构造函数有三个参数：

# optimizer：一个 PyTorch 优化器对象，如 torch.optim.SGD、torch.optim.Adam 等。
# step_size：降低学习率的间隔步数，即经过多少个迭代步骤后降低学习率。
# gamma：学习率降低的倍数，即学习率每次降低后的缩放比例。
# conf.patience初始化为50
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
best_loss = 100000		#这个参数什么意思，最佳损失值初始化为100000，这么大？
intial_epoch =0
num_epoch_no_improvement = 0	#这个参数什么意思
sys.stdout.flush()

if conf.weights != None:
	checkpoint=torch.load(conf.weights)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	intial_epoch=checkpoint['epoch']
	print("Loading weights from ",conf.weights)
sys.stdout.flush()

	#	conf.nb_epoch为设置的完整遍历次数，一个epoch意味着你已经训练了所有数据集（所有记录）一次
	#	epoch的值告诉模型它必须重复上述所有过程的次数，然后停止
for epoch in range(intial_epoch,conf.nb_epoch):
	scheduler.step(epoch)		#对学习率进行调整

	# 	在使用 pytorch 构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是 启用 batch normalization 和 dropout 。
	# 如果模型中有BN层（Batch Normalization）和 Dropout ，需要在 训练时 添加 model.train()。
	# model.train() 是保证 BN 层能够用到 每一批数据 的均值和方差。对于 Dropout，model.train() 是 随机取一部分 网络连接来训练更新参数。

	model.train()
	for iteration in range(int(x_train.shape[0]//conf.batch_size)):
		image, gt = next(training_generator)		#返回下一个元组(x,y)
		gt = np.repeat(gt,conf.nb_class,axis=1)		#这个nb.class啥意思,gt为图像标签

		# 		功能：torch.from_numpy(ndarray) → Tensor，即 从numpy.ndarray创建一个张量。
		# 说明：返回的张量和ndarray共享同一内存。对张量的修改将反映在ndarray中，反之亦然。返回的张量是不能调整大小的。

		image,gt = torch.from_numpy(image).float().to(device), torch.from_numpy(gt).float().to(device)
		pred=model(image)
		loss = criterion(pred,gt)		#gt为图像标签
		optimizer.zero_grad()			#梯度清零
		loss.backward()					#反向传播
		optimizer.step()				#更新模型参数

		#round(),四舍六入五留双
		# 要求保留位数的后一位<=4，则进位，如round(5.214,2)保留小数点后两位，结果是 5.21
		# 要求保留位数的后一位“=5”，且该位数后面没有数字，则不进位，如round(5.215,2)，结果为5.21
		# 要求保留位数的后一位“=5”，且该位数后面有数字，则进位，如round(5.2151,2)，结果为5.22
		# 要求保留位数的后一位“>=6”，则进位。如round(5.216,2)，结果为5.22
		# 1.item（）取出张量具体位置的元素元素值
		# 2.并且返回的是该位置元素值的高精度值

		train_losses.append(round(loss.item(), 2))
		if (iteration + 1) % 5 ==0:
			print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
				.format(epoch + 1, conf.nb_epoch, iteration + 1, np.average(train_losses)))
			sys.stdout.flush()		#输出缓冲区

	# requires_grad
	# 在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。
	# tensor的requires_grad的属性默认为False,若一个节点（叶子变量：自己创建的tensor）requires_grad被设置为True，那么所有依赖它的节点requires_grad都为True（即使其他相依赖的tensor的requires_grad = False）
	# 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
	# with torch.no_grad的作用
	# 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
	# 即使一个tensor（命名为x）的requires_grad = True，在with torch.no_grad计算，由x得到的新tensor（命名为w-标量）requires_grad也为False，且grad_fn也为None,即不会对w求导。
	with torch.no_grad():
		model.eval()
		# model.eval()的作用是 不启用 Batch Normalization 和 Dropout。
		# 如果模型中有 BN 层（Batch Normalization）和 Dropout，在 测试时 添加 model.eval()。
		# model.eval() 是保证 BN 层能够用 全部训练数据 的均值和方差，即测试过程中要保证 BN 层的均值和方差不变。
		# 对于 Dropout，model.eval() 是利用到了 所有 网络连接，即不进行随机舍弃神经元。
		print("validating....")
		for i in range(int(x_valid.shape[0]//conf.batch_size)):
			x,y = next(validation_generator)
			y = np.repeat(y,conf.nb_class,axis=1)
			image,gt = torch.from_numpy(x).float(), torch.from_numpy(y).float()
			image=image.to(device)
			gt=gt.to(device)
			pred=model(image)
			loss = criterion(pred,gt)
			valid_losses.append(loss.item())
	
	#logging
	train_loss=np.average(train_losses)
	valid_loss=np.average(valid_losses)
	avg_train_losses.append(train_loss)
	avg_valid_losses.append(valid_loss)
	print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
	train_losses=[]
	valid_losses=[]
	if valid_loss < best_loss:
		print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
		best_loss = valid_loss
		num_epoch_no_improvement = 0		#这个参数赋值为0是什么意思？
		#save model
		torch.save({
			'epoch': epoch+1,
			'state_dict' : model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()
		},os.path.join(conf.model_path, "Genesis_Chest_CT.pt"))
		print("Saving model ",os.path.join(conf.model_path,"Genesis_Chest_CT.pt"))
	else:
		print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
		num_epoch_no_improvement += 1		#如果一次epoch没改进就把num_epoch_no_improvement加1
		
	#将epochs设置为conf.patience，当验证精度或损失停止提高时停止训练：所谓的early stopping
	if num_epoch_no_improvement == conf.patience:
		print("Early Stopping")
		break
	sys.stdout.flush()
