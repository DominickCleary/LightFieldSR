import sys

import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
from utils import normalize, imsave, avg_msssim, psnr, un_normalize
from Model import SingleGenerator, SingleDiscriminator
import torch.optim as optim
import torch.nn as nn
import time

# modelv1 training


def train_single(generator, discriminator, opt, dataloader, writer, scale):
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    ones_const = Variable(torch.ones(1, 1))

    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()

    optimizer = optim.Adam(generator.parameters(), lr=opt.generatorLR)
    optim_discriminator = optim.Adam(
        discriminator.parameters(), lr=opt.discriminatorLR)

    scheduler_gen = ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3, verbose=True)
    scheduler_dis = ReduceLROnPlateau(
        optim_discriminator, 'min', factor=0.5, patience=3, verbose=True)

    curr_time = time.time()

    for epoch in range(opt.nEpochs):
        mean_generator_content_loss = 0.0
        mean_generator_adversarial_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0
        high_res_fake = 0

        for phase in ['train', 'test']:
            if phase == 'test':
                generator.train(False)
                discriminator.train(False)
            else:
                generator.train(True)
                discriminator.train(True)

            for batch_no, data in enumerate(dataloader[phase]):
                high_img, _ = data

                input1 = high_img[0, :, :, :]
                input2 = high_img[1, :, :, :]
                input3 = high_img[2, :, :, :]
                input4 = high_img[3, :, :, :]
                inputs = torch.FloatTensor(
                    opt.batchSize, 3, opt.imageSize, opt.imageSize)
                # imshow(input3)

                for j in range(opt.batchSize):
                    inputs[j] = scale(high_img[j])
                    high_img[j] = normalize(high_img[j])

                # save_image(high_img[3], "test.png")

                high_comb = torch.cat(
                    [high_img[0], high_img[1], high_img[2], high_img[3]], 0)

                high_comb = Variable(high_comb[np.newaxis, :]).cuda()
                # print(high_comb.cpu().data)
                input_comb = torch.cat([scale(input1), scale(
                    input2), scale(input3), scale(input4)], 0)
                input_comb = input_comb[np.newaxis, :]

                if opt.cuda:
                    optimizer.zero_grad()
                    high_res_real = Variable(high_img.cuda())
                    high_res_fake = generator(Variable(input_comb).cuda())
                    target_real = Variable(torch.rand(1, 1) * 0.5 + 0.7).cuda()
                    target_fake = Variable(torch.rand(1, 1) * 0.3).cuda()

                    outputs = torch.chunk(high_res_fake, 4, 1)
                    outputs = torch.cat(
                        [outputs[0], outputs[1], outputs[2], outputs[3]], 0)

                    discriminator.zero_grad()

                    discriminator_loss = adversarial_criterion(discriminator(high_comb), target_real) + \
                        adversarial_criterion(discriminator(
                            Variable(high_res_fake.data)), target_fake)
                    mean_discriminator_loss += discriminator_loss.data.item()

                    if phase == 'train':
                        discriminator_loss.backward()
                        optim_discriminator.step()

                    generator_content_loss = content_criterion(
                        high_res_fake, high_comb)
                    mean_generator_content_loss += generator_content_loss.data.item()
                    generator_adversarial_loss = adversarial_criterion(
                        discriminator(high_res_fake), ones_const)
                    mean_generator_adversarial_loss += generator_adversarial_loss.data.item()

                    generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
                    mean_generator_total_loss += generator_total_loss.data.item()
                    # print(mean_generator_total_loss)
                    # time.sleep(10)

                    if phase == 'train':
                        generator_total_loss.backward()
                        optimizer.step()

                    if (batch_no % 10 == 0):
                        sys.stdout.write(
                            '\rphase [%s] epoch [%d/%d] batch no. [%d/%d] Generator_content_Loss: %.4f Discriminator loss: %.4f' % (
                                phase, epoch, opt.nEpochs, batch_no, len(dataloader[phase]), generator_content_loss, discriminator_loss))

            # imshow(high_res_fake.cpu().data)
            scheduler_gen.step(mean_generator_total_loss)
            scheduler_dis.step(mean_discriminator_loss)
            psnr_val = psnr(un_normalize(high_res_real), un_normalize(outputs))
            # imsave(outputs.cpu().data, train=True, epoch=epoch, image_type='fake')
            # imsave(high_img, train=True, epoch=epoch, image_type='real')
            # imsave(inputs, train=True, epoch=epoch, image_type='low')

            writer.add_scalar(phase+" per epoch/generator lr",
                              optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar(phase+" per epoch/discriminator lr",
                              optim_discriminator.param_groups[0]['lr'], epoch)
            writer.add_scalar(phase + " per epoch/PSNR", psnr_val, epoch)
            if phase == 'train':
                writer.add_scalar(
                    phase+" per epoch/discriminator training loss", mean_discriminator_loss, epoch)
                writer.add_scalar(
                    phase+" per epoch/generator training loss", mean_generator_total_loss, epoch)

            writer.add_scalar("per epoch/time taken",
                              time.time()-curr_time, epoch)
            torch.save(generator.state_dict(),
                       '%s/generator_single.pth' % opt.out)
            torch.save(discriminator.state_dict(),
                       '%s/discriminator_single.pth' % opt.out)


# Training of simple CNN model model.
def train_firstmodel(generator, opt, dataloader, writer, scale):
    content_criterion = nn.MSELoss()

    ones_const = Variable(torch.ones(1, 1))

    if opt.cuda:
        generator.cuda()
        content_criterion.cuda()

    optimizer = optim.SGD(generator.parameters(), lr=opt.generatorLR)
    scheduler_gen = ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3, verbose=True)
    curr_time = time.time()

    for epoch in range(opt.nEpochs):
        mean_generator_content_loss = 0.0
        mean_generator_total_loss = 0.0

        high_res_fake = 0
        for phase in ['train', 'test']:

            if phase == 'test':
                generator.train(False)
            else:
                generator.train(True)

            for batch_no, data in enumerate(dataloader[phase]):
                high_img, _ = data

                input1 = high_img[0, :, :, :]
                input2 = high_img[1, :, :, :]
                input3 = high_img[2, :, :, :]
                input4 = high_img[3, :, :, :]
                # imshow(input3)
                for j in range(opt.batchSize):
                    high_img[j] = normalize(high_img[j])
                high_comb = torch.cat(
                    [high_img[0], high_img[1], high_img[2], high_img[3]], 0)

                high_comb = Variable(high_comb[np.newaxis, :]).cuda()
                # imshow(high_comb.cpu().data)
                input_comb = torch.cat([scale(input1), scale(
                    input2), scale(input3), scale(input4)], 0)
                input_comb = input_comb[np.newaxis, :]
                if opt.cuda:
                    if phase == 'train':
                        optimizer.zero_grad()
                    high_res_real = Variable(high_img.cuda())
                    high_res_fake = generator(Variable(input_comb).cuda())

                    outputs = torch.chunk(high_res_fake, 4, 1)
                    outputs = torch.cat(
                        [outputs[0], outputs[1], outputs[2], outputs[3]], 0)
                    # imshow(outputs[0])
                    generator_content_loss = content_criterion(
                        high_res_fake, high_comb)
                    mean_generator_content_loss += generator_content_loss.data[0]

                    generator_total_loss = generator_content_loss
                    mean_generator_total_loss += generator_total_loss.data[0]

                    if phase == 'train':
                        generator_total_loss.backward()
                        optimizer.step()

                    if (batch_no % 10 == 0):
                        sys.stdout.write('\rphase [%s] epoch [%d/%d] batch no. [%d/%d] Generator_content_Loss: %.4f ' % (
                            phase, epoch, opt.nEpochs, batch_no, len(dataloader[phase]), generator_content_loss))

            if phase == 'train':
                # imsave(outputs,train=True,epoch=epoch,image_type='fake')
                # imsave(high_img, train=True, epoch=epoch, image_type='real')
                # imsave(input_comb, train=True, epoch=epoch, image_type='low')
                writer.add_scalar(phase + " per epoch/generator lr",
                                  optimizer.param_groups[0]['lr'], epoch + 1)
                scheduler_gen.step(
                    mean_generator_total_loss / len(dataloader[phase]))

            mssim = avg_msssim(high_res_real, outputs)
            psnr_val = psnr(un_normalize(high_res_real), un_normalize(outputs))

            writer.add_scalar(phase + " per epoch/PSNR", psnr_val,
                              epoch + 1)
            writer.add_scalar(phase+" per epoch/generator loss",
                              mean_generator_total_loss/len(dataloader[phase]), epoch+1)
            writer.add_scalar("per epoch/total time taken",
                              time.time()-curr_time, epoch+1)
            writer.add_scalar(phase+" per epoch/avg_mssim", mssim, epoch+1)

            torch.save(generator.state_dict(),
                       '%s/generator_firstfinal.pth' % opt.out)
