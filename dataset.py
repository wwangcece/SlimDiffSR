import torch, random, cv2, os, math, glob
import torch.nn.functional as F
import numpy as np
from bsr.degradations import (
    circular_lowpass_kernel,
    random_mixed_kernels,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from bsr.transforms import augment, paired_random_crop
from bsr.utils import FileClient, imfrombytes, img2tensor, DiffJPEG
from bsr.utils.img_process_util import filter2D


class RealESRGANDataset(torch.utils.data.Dataset):
    def __init__(self, opt, bsz):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = FileClient("disk")
        self.gt_folder = opt["dataroot_gt"]
        self.len = bsz * opt["iter_num"]
        self.paths = glob.glob(os.path.join(self.gt_folder, "**/*"), recursive=True)

        # blur settings for the first degradation
        self.blur_kernel_size = opt["blur_kernel_size"]
        self.kernel_list = opt["kernel_list"]
        self.kernel_prob = opt["kernel_prob"]  # a list for each kernel probability
        self.blur_sigma = opt["blur_sigma"]
        self.betag_range = opt[
            "betag_range"
        ]  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt["betap_range"]  # betap used in plateau blur kernels
        self.sinc_prob = opt["sinc_prob"]  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt["blur_kernel_size2"]
        self.kernel_list2 = opt["kernel_list2"]
        self.kernel_prob2 = opt["kernel_prob2"]
        self.blur_sigma2 = opt["blur_sigma2"]
        self.betag_range2 = opt["betag_range2"]
        self.betap_range2 = opt["betap_range2"]
        self.sinc_prob2 = opt["sinc_prob2"]

        # a final sinc filter
        self.final_sinc_prob = opt["final_sinc_prob"]

        self.kernel_range = [
            2 * v + 1 for v in range(3, 11)
        ]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        index = random.randint(0, len(self.paths) - 1)
        gt_path = self.paths[index]
        img_gt = imfrombytes(self.file_client.get(gt_path, "gt"), float32=True)
        img_gt = augment(img_gt, self.opt["use_hflip"], self.opt["use_rot"])
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.opt.gt_size
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(
                img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top : top + crop_pad_size, left : left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob"]:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob2"]:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt["final_sinc_prob"]:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {
            "gt": img_gt,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "gt_path": gt_path,
        }
        return return_d

    def __len__(self):
        return self.len


class RealESRGANDegrader:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.jpeger = DiffJPEG(differentiable=False).to(
            device
        )  # simulate JPEG compression artifacts
        self.queue_size = 1200

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            assert (
                self.queue_size % b == 0
            ), f"queue size {self.queue_size} should be divisible by batch size {b}"
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).to(self.device)
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).to(self.device)
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.lq.clone()
            )
            self.queue_gt[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.gt.clone()
            )
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def degrade(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images."""
        # training data synthesis
        self.gt = data["gt"].to(self.device)

        self.kernel1 = data["kernel1"].to(self.device)
        self.kernel2 = data["kernel2"].to(self.device)
        self.sinc_kernel = data["sinc_kernel"].to(self.device)

        ori_h, ori_w = self.gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(self.gt, self.kernel1)
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.opt["resize_prob"])[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.opt["resize_range"][1])
        elif updown_type == "down":
            scale = np.random.uniform(self.opt["resize_range"][0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.opt["gray_noise_prob"]
        if np.random.uniform() < self.opt["gaussian_noise_prob"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.opt["noise_range"],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt["poisson_scale_range"],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
        out = torch.clamp(
            out, 0, 1
        )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt["second_blur_prob"]:
            out = filter2D(out, self.kernel2)
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.opt["resize_prob2"])[
            0
        ]
        if updown_type == "up":
            scale = np.random.uniform(1, self.opt["resize_range2"][1])
        elif updown_type == "down":
            scale = np.random.uniform(self.opt["resize_range2"][0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out,
            size=(
                int(ori_h / self.opt["scale"] * scale),
                int(ori_w / self.opt["scale"] * scale),
            ),
            mode=mode,
        )
        # add noise
        gray_noise_prob = self.opt["gray_noise_prob2"]
        if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.opt["noise_range2"],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt["poisson_scale_range2"],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(
                out,
                size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                mode=mode,
            )
            out = filter2D(out, self.sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(
                out,
                size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                mode=mode,
            )
            out = filter2D(out, self.sinc_kernel)

        # clamp and round
        self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        # random crop
        gt_size = self.opt["gt_size"]
        self.gt, self.lq = paired_random_crop(
            self.gt, self.lq, gt_size, self.opt["scale"]
        )

        # training pair pool
        self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        self.lq = (
            self.lq.contiguous()
        )  # for the warning: grad and param do not obey the gradient layout contract

        return self.lq, self.gt


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(PairedDataset, self).__init__()
        self.opt = opt
        self.file_client = FileClient("disk")
        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt["dataroot_lq"]
        self.gt_paths = glob.glob(os.path.join(self.gt_folder, "**/*"), recursive=True)
        self.lq_paths = glob.glob(os.path.join(self.lq_folder, "**/*"), recursive=True)

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        lq_path = self.lq_paths[index]
        img_gt = imfrombytes(self.file_client.get(gt_path, "gt"), float32=True)
        img_lq = imfrombytes(self.file_client.get(lq_path, "lq"), float32=True)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return_d = {"gt": img_gt, "gt_path": gt_path, "lq": img_lq}
        return return_d

    def __len__(self):
        return len(self.gt_paths)
