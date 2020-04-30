import torch
import torchvision
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

from torchvision.transforms import transforms, Normalize
from torchvision.utils import save_image, make_grid

from utils import normalize, imsave, avg_msssim, psnr, psnr_batch, un_normalize
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Model import FeatureExtractor, InferAesthetic
import sys
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

aesthetic_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def bilinear_upsampling(opt, dataloader, scale):
    for batch_no, data in enumerate(dataloader['test']):
        high_img, _ = data
        inputs = torch.FloatTensor(
            opt.batchSize, 3, opt.imageSize, opt.imageSize)

        for j in range(opt.batchSize):
            inputs[j] = scale(high_img[j])
            high_img[j] = normalize(high_img[j])

        outputs = F.upsample(inputs, scale_factor=opt.upSampling,
                             mode='bilinear', align_corners=True)
        transform = transforms.Compose([transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444]),
                                        transforms.ToPILImage()])
        transform(outputs[0]).save(
            'output/train/bilinear_fake/' + str(batch_no) + '.png')
        transform(high_img[0]).save(
            'output/train/bilinear_real/' + str(batch_no) + '.png')

        # for output, himg in zip (outputs, high_img):
        #     psnr_val = psnr(output,himg)
        #mssim = avg_msssim(himg, output)
        print(psnr(un_normalize(outputs), un_normalize(high_img)))

# normal training


def train_multiple(generator, discriminator, opt, dataloader, writer, scale):
    feature_extractor = FeatureExtractor(
        torchvision.models.vgg19(pretrained=True))

    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()
    aesthetic_loss = AestheticLoss()

    ones_const = Variable(torch.ones(opt.batchSize, 1))

    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        feature_extractor.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()

    optimizer = optim.Adam(generator.parameters(), lr=opt.generatorLR)
    optim_discriminator = optim.Adam(
        discriminator.parameters(), lr=opt.discriminatorLR)
    scheduler_gen = ReduceLROnPlateau(
        optimizer, 'min', factor=0.7, patience=10, verbose=True)
    scheduler_dis = ReduceLROnPlateau(
        optim_discriminator, 'min', factor=0.7, patience=10, verbose=True)
    curr_time = time.time()
    inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

    # pretraining
    for epoch in range(2):
        mean_generator_content_loss = 0.0

        inputs = torch.FloatTensor(
            opt.batchSize, 3, opt.imageSize, opt.imageSize)

        for batch_no, data in enumerate(dataloader['train']):
            high_img, _ = data
            # save_image(high_img, "test.png")
            # time.sleep(10)

            for j in range(opt.batchSize):
                inputs[j] = scale(high_img[j])
                high_img[j] = normalize(high_img[j])

            # print(high_img[0].shape)
            # print(inputs[0].shape)
            # time.sleep(10)

            if opt.cuda:
                optimizer.zero_grad()
                high_res_real = Variable(high_img.cuda())
                high_res_fake = generator(Variable(inputs[0][np.newaxis, :]).cuda(), Variable(inputs[1][np.newaxis, :]).cuda(
                ), Variable(inputs[2][np.newaxis, :]).cuda(), Variable(inputs[3][np.newaxis, :]).cuda())

                generator_content_loss = content_criterion(
                    high_res_fake, high_res_real)

                mean_generator_content_loss += generator_content_loss.data.item()

                generator_content_loss.backward()
                optimizer.step()

                sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (
                    epoch, 2, batch_no, len(dataloader['train']), generator_content_loss.data.item()))

    # training
    for epoch in range(opt.nEpochs):
        for phase in ['train', 'test']:
            if phase == 'test':
                generator.train(False)
                discriminator.train(False)
            else:
                generator.train(True)
                discriminator.train(True)

            mean_generator_content_loss = 0.0
            mean_generator_adversarial_loss = 0.0
            mean_generator_total_loss = 0.0
            mean_discriminator_loss = 0.0
            # mean_psnr = 0.0
            # mean_msssim = 0.0
            high_img = torch.FloatTensor(
                opt.batchSize, 3, opt.imageSize, opt.imageSize)
            inputs = torch.FloatTensor(
                opt.batchSize, 3, opt.imageSize, opt.imageSize)

            for batch_no, data in enumerate(dataloader[phase]):
                high_img, _ = data

                for j in range(opt.batchSize):
                    inputs[j] = scale(high_img[j])
                    high_img[j] = normalize(high_img[j])

                if opt.cuda:
                    optimizer.zero_grad()
                    high_res_real = Variable(high_img.cuda())
                    high_res_fake = generator(Variable(inputs[0][np.newaxis, :]).cuda(), Variable(inputs[1][np.newaxis, :]).cuda(
                    ), Variable(inputs[2][np.newaxis, :]).cuda(), Variable(inputs[3][np.newaxis, :]).cuda())

                    # save_image(high_res_real, "REAL.png")
                    # save_image(high_res_fake, "FAKE.png")

                    target_real = Variable(torch.rand(
                        opt.batchSize, 1) * 0.5 + 0.7).cuda()
                    target_fake = Variable(torch.rand(
                        opt.batchSize, 1) * 0.3).cuda()

                    discriminator.zero_grad()

                    discriminator_loss = adversarial_criterion(
                        discriminator(Variable(inputs[0][np.newaxis, :]).cuda(), Variable(inputs[1][np.newaxis, :]).cuda(),
                                      Variable(inputs[2][np.newaxis, :]).cuda(), Variable(inputs[3][np.newaxis, :]).cuda()),
                        target_real) + \
                        adversarial_criterion(
                        discriminator(high_res_fake[0][np.newaxis, :], high_res_fake[1][np.newaxis, :], high_res_fake[2][np.newaxis, :],
                                      high_res_fake[3][np.newaxis, :]), target_fake)
                    mean_discriminator_loss += discriminator_loss.data.item()

                    if phase == 'train':
                        discriminator_loss.backward(retain_graph=True)
                        optim_discriminator.step()

                    #high_res_fake_cat = torch.cat([ image for image in high_res_fake ], 0)
                    fake_features = feature_extractor(high_res_fake)
                    real_features = Variable(
                        feature_extractor(high_res_real).data)

                    # generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
                    generator_content_loss = content_criterion(high_res_fake,
                                                               high_res_real) + content_criterion(fake_features,
                                                                                                  real_features)
                    mean_generator_content_loss += generator_content_loss.data.item()
                    generator_adversarial_loss = adversarial_criterion(discriminator(
                        high_res_fake[0][np.newaxis, :], high_res_fake[1][np.newaxis, :], high_res_fake[2][np.newaxis, :], high_res_fake[3][np.newaxis, :]), ones_const)
                    mean_generator_adversarial_loss += generator_adversarial_loss.data.item()

                    generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
                    mean_generator_total_loss += generator_total_loss.data.item()

                    if phase == 'train':
                        generator_total_loss.backward()
                        optimizer.step()

                    if(batch_no % 10 == 0):
                        # print("phase {} batch no. {} generator_content_loss {} discriminator_loss {}".format(phase, batch_no, generator_content_loss, discriminator_loss))
                        sys.stdout.write('\rphase [%s] epoch [%d/%d] batch no. [%d/%d] Generator_content_Loss: %.4f discriminator_loss %.4f' % (
                            phase, epoch, opt.nEpochs, batch_no, len(dataloader[phase]), generator_content_loss, discriminator_loss))

            if phase == 'train':
                imsave(high_res_fake.cpu().data, train=True,
                       epoch=epoch, image_type='fake')
                imsave(high_img, train=True, epoch=epoch, image_type='real')
                imsave(inputs, train=True, epoch=epoch, image_type='low')
                writer.add_scalar(phase + " per epoch/generator lr",
                                  optimizer.param_groups[0]['lr'], epoch + 1)
                writer.add_scalar(phase + " per epoch/discriminator lr", optim_discriminator.param_groups[0]['lr'],
                                  epoch + 1)
                scheduler_gen.step(
                    mean_generator_total_loss / len(dataloader[phase]))
                scheduler_dis.step(mean_discriminator_loss /
                                   len(dataloader[phase]))
            else:
                imsave(high_res_fake.cpu().data, train=False,
                       epoch=epoch, image_type='fake')
                imsave(high_img, train=False, epoch=epoch, image_type='real')
                imsave(inputs, train=False, epoch=epoch, image_type='low')
            # import ipdb;
            # ipdb.set_trace()
            mssim = avg_msssim(high_res_real, high_res_fake)
            psnr_val = psnr(un_normalize(high_res_real),
                            un_normalize(high_res_fake))

            writer.add_scalar(phase + " per epoch/PSNR", psnr_val,
                              epoch + 1)
            writer.add_scalar(phase+" per epoch/discriminator loss",
                              mean_discriminator_loss/len(dataloader[phase]), epoch+1)
            writer.add_scalar(phase+" per epoch/generator loss",
                              mean_generator_total_loss/len(dataloader[phase]), epoch+1)
            writer.add_scalar("per epoch/total time taken",
                              time.time()-curr_time, epoch+1)
            writer.add_scalar(phase+" per epoch/avg_mssim", mssim, epoch+1)
        # Do checkpointing
        torch.save(generator.state_dict(), '%s/generator_final.pth' % opt.out)
        torch.save(discriminator.state_dict(),
                   '%s/discriminator_final.pth' % opt.out)

# for model Angresv2. Will require matlab pre-processed images.


def train_angres(AngRes, lflists, opt, writer):
    content_criterion = nn.MSELoss()
    aesthetic_criterion = AestheticLoss()

    # fake_ang_res = torch.FloatTensor(
    #     4, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)
    optimizer = optim.Adam(AngRes.parameters(), lr=opt.angResLR)
    scheduler_angres = ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=3, verbose=True)

    curr_time = time.time()

    if opt.cuda:
        AngRes.cuda()
        content_criterion.cuda()

    AngRes.train(True)

    for epoch in range(opt.continue_from, opt.nEpochs):
        mean_loss = 0.0
        count = 0
        for lf_image in lflists:
            i = j = -1
            new_img = torch.FloatTensor(
                4, 3, opt.upSampling*opt.imageSize, opt.upSampling*opt.imageSize)

            while i < 14:
                i += 1
                j = -1
                while j < 14:
                    j += 1
                    img1 = torch.Tensor(lf_image[i][j])
                    img2 = torch.Tensor(lf_image[i][j+2])
                    img3 = torch.Tensor(lf_image[i+2][j])
                    img4 = torch.Tensor(lf_image[i+2][j+2])
                    gt_center = torch.Tensor(lf_image[i+1][j+1])
                    gt_horizontaltop = torch.Tensor(lf_image[i][j + 1])
                    gt_verticalleft = torch.Tensor(lf_image[i + 1][j])
                    gt_horizontalbottom = torch.Tensor(lf_image[i + 2][j + 1])
                    gt_verticalright = torch.Tensor(lf_image[i + 1][j + 2])

                    img1 = torch.transpose(img1, 0, 2)
                    img2 = torch.transpose(img2, 0, 2)
                    img3 = torch.transpose(img3, 0, 2)
                    img4 = torch.transpose(img4, 0, 2)

                    gt_center = torch.transpose(gt_center, 0, 2)
                    gt_horizontaltop = torch.transpose(gt_horizontaltop, 0, 2)
                    gt_verticalleft = torch.transpose(gt_verticalleft, 0, 2)
                    gt_horizontalbottom = torch.transpose(
                        gt_horizontalbottom, 0, 2)
                    gt_verticalright = torch.transpose(gt_verticalright, 0, 2)

                    new_img[0] = torch.transpose(img1, 1, 2)
                    new_img[1] = torch.transpose(img2, 1, 2)
                    new_img[2] = torch.transpose(img3, 1, 2)
                    new_img[3] = torch.transpose(img4, 1, 2)

                    gt_center_img = torch.transpose(
                        gt_center, 1, 2).type(torch.FloatTensor)
                    gt_horizontaltop_img = torch.transpose(
                        gt_horizontaltop, 1, 2).type(torch.FloatTensor)
                    gt_verticalleft_img = torch.transpose(
                        gt_verticalleft, 1, 2).type(torch.FloatTensor)
                    gt_horizontalbottom_img = torch.transpose(
                        gt_horizontalbottom, 1, 2).type(torch.FloatTensor)
                    gt_verticalright_img = torch.transpose(
                        gt_verticalright, 1, 2).type(torch.FloatTensor)

                    if opt.cuda:
                        fake_img = AngRes(Variable(new_img[0][np.newaxis, :]).cuda(),
                                          Variable(new_img[1][np.newaxis, :]).cuda(),
                                          Variable(new_img[2][np.newaxis, :]).cuda(),
                                          Variable(new_img[3][np.newaxis, :]).cuda())
                      
                        out = [aesthetic_transform(fake_img[0].cpu().data.clone()).cuda(),
                               aesthetic_transform(fake_img[1].cpu().data.clone()).cuda(),
                               aesthetic_transform(fake_img[2].cpu().data.clone()).cuda(),
                               aesthetic_transform(fake_img[3].cpu().data.clone()).cuda(),
                               aesthetic_transform(fake_img[4].cpu().data.clone()).cuda()]

                        target = [aesthetic_transform(gt_center_img.clone()).cuda(),
                                  aesthetic_transform(gt_horizontaltop_img.clone()).cuda(),
                                  aesthetic_transform(gt_horizontalbottom_img.clone()).cuda(),
                                  aesthetic_transform(gt_verticalleft_img.clone()).cuda(),
                                  aesthetic_transform(gt_verticalright_img.clone()).cuda()]

                        aesthetic_loss = aesthetic_criterion(out, target)
                        
                        center_loss = content_criterion(
                            fake_img[0], gt_center_img.cuda())
                        horizontaltop_loss = content_criterion(
                            fake_img[1], gt_horizontaltop_img.cuda())
                        horizontalbottom_loss = content_criterion(
                            fake_img[2], gt_horizontalbottom_img.cuda())
                        verticalleft_loss = content_criterion(
                            fake_img[3], gt_verticalleft_img.cuda())
                        verticalright_loss = content_criterion(
                            fake_img[4], gt_verticalright_img.cuda())
                        
                        total_loss = center_loss + horizontaltop_loss + \
                            horizontalbottom_loss + verticalleft_loss + verticalright_loss + (aesthetic_loss * 1e5)
                        
                        # print()
                        # print(total_loss)
                        mean_loss += total_loss
                        AngRes.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        if (j == 14):
                            if(opt.progress_images):
                                images = []
                                for x in range(0, 5):
                                    images.append(display_transform(fake_img[x].cpu().data.clone().type(torch.ByteTensor)))

                                image = make_grid(images, padding=1)
                                save_image(image, "fake.png")

                                one = display_transform(
                                    gt_center_img.clone().type(torch.ByteTensor))
                                two = display_transform(
                                    gt_horizontaltop_img.clone().type(torch.ByteTensor))
                                three = display_transform(
                                    gt_horizontalbottom_img.clone().type(torch.ByteTensor))
                                four = display_transform(
                                    gt_verticalleft_img.clone().type(torch.ByteTensor))
                                five = display_transform(
                                    gt_verticalright_img.clone().type(torch.ByteTensor))

                                real_image = make_grid(
                                    [one, two, three, four, five], padding=1)
                                save_image(real_image, "real.png")

                            sys.stdout.write(
                                '\repoch [%d/%d][%.2f%%] content_Loss: %.4f' % (
                                    epoch, opt.nEpochs, ((count * 14) + i)/(14 * len(lflists)) * 100, total_loss))

            imsave(fake_img.cpu().data, train=True, epoch=count,
                   image_type='new', ang_res=True, row_ind=i, column_ind=j)
            imsave(gt_center_img, train=True, epoch=count,
                   image_type='real_center', ang_res=True)
            imsave(gt_horizontaltop_img, train=True, epoch=count,
                   image_type='real_horizontaltop', ang_res=True)
            imsave(gt_horizontalbottom_img, train=True, epoch=count,
                   image_type='real_horizontalbottom', ang_res=True)
            imsave(gt_verticalleft_img, train=True, epoch=count,
                   image_type='real_verticalleft', ang_res=True)
            imsave(gt_verticalright_img, train=True, epoch=count,
                   image_type='real_verticalright', ang_res=True)
            writer.add_scalar(" per epoch/angres lr",
                              optimizer.param_groups[0]['lr'], epoch + 1)
            count += 1
            # else:
            #     imsave(fake_img.cpu().data, train=False, epoch=epoch, image_type='new', ang_res=True)
            #     imsave(inputs, train=False, epoch=epoch, image_type='real', ang_res=True)
        scheduler_angres.step(total_loss / len(lflists))
        writer.add_scalar(" per epoch/angres loss", mean_loss / len(lflists),
                          epoch + 1)
        writer.add_scalar("per epoch/total time taken",
                          time.time() - curr_time, epoch + 1)

        # Do checkpointing
        torch.save(AngRes.state_dict(), '%s/AngRes_final.pth' % opt.out)

# Aesthetic Loss class, based on NIMA


class AestheticLoss(nn.Module):
    def __init__(self):
        super(AestheticLoss, self).__init__()
        self.aesthetic_loss = InferAesthetic()

    def forward(self, out, target):

        # print()
        # time.sleep(10)
        fake_predict = torch.Tensor(len(out)).to(DEVICE)
        target_predict = torch.Tensor(len(target)).to(DEVICE)

        # print(out[x].shape)
        # time.sleep(10)
        for x in range(0, len(out)):
            fake_predict[x] = self.aesthetic_loss(out[x])
            
        # print(fake_predict)
        # time.sleep(10)
        # print("FAKE" + str(fake_predict[x]))
            target_predict[x] = self.aesthetic_loss(target[x])
            # print(fake_predict[x])
            # time.sleep(1)
        # print(target_predict)
        # print()
        # print("REAL" + str(target_predict[x]))

        fake_mean = torch.mean(fake_predict)
        target_mean = torch.mean(target_predict)
        # Get the difference between average values of the the target and fake images
        result = torch.abs(target_mean - fake_mean)
        # print(result)
        return result
