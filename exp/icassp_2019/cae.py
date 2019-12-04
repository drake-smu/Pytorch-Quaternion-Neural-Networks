import torch
from   torch                import nn
from   convolutional_models import CAE, QCAE, QAE

import os
from   imageio import imread,imwrite
import numpy   as np
import sys


def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


MODEL         = str(sys.argv[1])
CUDA          = False 
NUM_EPOCHS    = 3001
LEARNING_RATE = 0.0005

if MODEL == 'QCAE':
    net  = QCAE()
else:
    net  = CAE()

#if CUDA:
#    net = net.CUDA()

#
# LOAD PICTURE
#

os.system('rm -rf data/save_image*')
if not os.path.isdir('out'):
	os.system('mkdir out')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

train = rgb2gray(imread('data/kodak/kodim05.png'))
imwrite("out/save_image_gray_training.png", train)
imwrite("out/girl_image_gray_training.png", rgb2gray(imread('data/kodak/kodim04.png')))
imwrite("out/parrot_image_gray_training.png",rgb2gray(imread('data/kodak/kodim23.png')))
train = train / 255

test = imread('data/kodak/kodim04.png')
test = test / 255

test2 = imread('data/kodak/kodim23.png')
test2 = test2 / 255
nb_param = sum(p.numel() for p in net.parameters() if p.requires_grad)

print("QCAE & CAE Color images - Titouan Parcollet - LIA, ORKIS")
print("Model Info --------------------")
print("Number of trainable parameters : "+str(nb_param))

if MODEL == 'QCAE':
    npad  = ((0, 0), (0, 0), (1, 0))
    train = np.pad(train, pad_width=npad, mode='constant', constant_values=0)
    train = np.transpose(train, (2,0,1))
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))


    test = np.pad(test, pad_width=npad, mode='constant', constant_values=0)
    test = np.transpose(test, (2,0,1))
    test = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))

    test2 = np.pad(test2, pad_width=npad, mode='constant', constant_values=0)
    test2 = np.transpose(test2, (2,0,1))
    test2 = np.reshape(test2, (1, test2.shape[0], test2.shape[1], test2.shape[2]))
else:
    train = np.transpose(train, (2,0,1))
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))

    test  = np.transpose(test, (2,0,1))
    test  = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))

    test2  = np.transpose(test2, (2,0,1))
    test2  = np.reshape(test2, (1, test2.shape[0], test2.shape[1], test2.shape[2]))

if CUDA:
    train = torch.from_numpy(train).float().CUDA()
    test  = torch.from_numpy(test).float().CUDA()
    test2  = torch.from_numpy(test2).float().CUDA()
else:
    train = torch.from_numpy(train).float()
    test  = torch.from_numpy(test).float()
    test2  = torch.from_numpy(test2).float()

for epoch in range(NUM_EPOCHS):

    output = net(train)
    loss   = criterion(output, train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("epoch "+str(epoch+1)+", loss_train "+str(loss.cpu().item()))

    if (epoch %100) == 0:

        output = net(test)
        out    = output.cpu().data.numpy()
        if MODEL == 'QCAE':
            out = np.transpose(out, (0,2,3,1))[:,:,:,1:]
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))
        else:
            out = np.transpose(out, (0,2,3,1))
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))


        imwrite("out/save_image_"+str(MODEL)+"_"+str(epoch)+".png", out)

validate = net(test2)
val_loss = criterion(validate,test2)
val_loss.backward()
val  = validate.cpu().data.numpy()

if MODEL == 'QCAE':
    val = np.transpose(val, (0,2,3,1))[:,:,:,1:]
    val = np.reshape(val, (val.shape[1], val.shape[2], val.shape[3]))
else:
    val = np.transpose(val, (0,2,3,1))
    val = np.reshape(val, (val.shape[1],val.shape[2],val.shape[3]))
print("#"*50+"\nFinal Eval",str(val_loss.cpu().item()));
imwrite("out/test_out_"+str(MODEL)+".png",val)
imwrite("out/test_in_"+str(MODEL)+".png",imread('data/kodak/kodim23.png'))
