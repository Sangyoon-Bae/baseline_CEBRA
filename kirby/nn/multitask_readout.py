from typing import Dict, List, Optional, Tuple, Union
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from kirby.taxonomy import DecoderSpec, Decoder, Task
from kirby.data.collate import collate, chain, track_batch
from kirby.nn import compute_loss_or_metric, UNet, UNetFiLM, HalfUNet, MLP
from kirby.nn.loss import * #PerceptualLoss, AlexNetPerceptualLoss, TVLoss, SSIMLoss, MultiScaleLoss, FocalLoss


class MultitaskReadout(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        decoder_specs: Dict[str, DecoderSpec],
        batch_type="stacked",
        use_cre_loss=False,
        use_random_masking_loss=False,
        no_decoding=False,
        masking_rate=0.5,
        pretrain=True,
        finetune=False,
        task='movie_decoding',
        unet_type='unet',
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_cre_loss = use_cre_loss
        self.use_random_masking_loss = use_random_masking_loss
        self.masking_rate = masking_rate
        self.no_decoding = no_decoding
        self.pretrain = pretrain
        self.finetune = finetune
        self.task = task
        self.unet_type = unet_type
        
        # Create a bunch of projection layers. One for each task
        self.projections = nn.ModuleDict({})
        if self.task in ['static_gratings', 'drifting_gratings']:
            for decoder_id, spec in decoder_specs.items():
                '''
                CRE_LINE DecoderSpec(dim=13, type=<OutputType.MULTINOMIAL: 3>, loss_fn='bce',
                timestamp_key='drifting_gratings.timestamps', value_key='drifting_gratings.orientation',
                task_key=None, subtask_key=None)
                '''
                if self.use_cre_loss and self.pretrain and decoder_id == 'CRE_LINE':
                    self.projections[decoder_id] = nn.Linear(latent_dim, 3)
                elif self.use_cre_loss and self.finetune and decoder_id == 'CRE_LINE':
                    self.projections[decoder_id] = nn.Linear(latent_dim, 10)
                else:
                    self.projections[decoder_id] = nn.Linear(latent_dim, spec.dim)

            
        # Need task specs layer to decide loss type
        self.decoder_specs = decoder_specs


        if 'movie_decoding' in self.task:
            if self.unet_type == 'unet':
                self.projections['NATURAL_MOVIE_RECONSTRUCTION'] = UNet(1, 1) # gray scale
            elif self.unet_type == 'unet_film':
                self.projections['NATURAL_MOVIE_RECONSTRUCTION'] = UNetFiLM(1, 1) # gray scale
            elif self.unet_type == 'half_unet':
                self.projections['NATURAL_MOVIE_RECONSTRUCTION'] = HalfUNet(1, 1, self.latent_dim) # gray scale
            elif self.unet_type == 'mlp':
                self.projections['NATURAL_MOVIE_RECONSTRUCTION'] = MLP(1, 1, self.latent_dim) # gray scale
        elif self.task == 'scene_decoding':
            if self.unet_type == 'unet':
                self.projections['NATURAL_SCENE_RECONSTRUCTION'] = UNet(1, 1) # gray scale
            elif self.unet_type == 'unet_film':
                self.projections['NATURAL_SCENE_RECONSTRUCTION'] = UNetFiLM(1, 1) # gray scale
            elif self.unet_type == 'half_unet':
                self.projections['NATURAL_SCENE_RECONSTRUCTION'] = HalfUNet(1, 1, self.latent_dim) # gray scale

        self.batch_type = batch_type
        

    def forward(
        self,
        output_latents: Union[
            TensorType["batch", "max_ntout", "dim"], TensorType["total_ntout", "dim"]
        ],
        output_decoder_index: Union[
            TensorType["batch", "max_ntout"], TensorType["total_ntout"]
        ],
        output_batch_index: Optional[TensorType["total_ntout"]] = None,
        output_values: Dict[str, TensorType["*ntout_task", "*nchannelsout"]] = None,
        output_weights: Dict[str, TensorType["*ntout_task"]] = None,
        cre_lines : TensorType["batch", "*nunits"] = None, # n_units = kind of 4-digits (Stella did it)
        masked_latents : TensorType["batch", "max_ntout", "dim"] = None,  # (Stella did it)
        original_latents : TensorType["batch", "max_ntout", "dim"] = None, # (Stella did it)
        unpack_output: bool = False,
        movie_frames : TensorType["batch", 304, 608] = None, # (Stella did it)
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        Union[None, torch.Tensor],
        Union[None, Dict[str, torch.Tensor]],
    ]:
        """
        Args:
            output_latents: Outputs of the last transformer layer.
            output_task_indices: Task index for each token in (batch, max_ntout).
            output_values: Ground-truth values for loss computation.
                output_values[task] is the ground truth value for the task
            output_weights: Sample-wise weights for loss computation.
                output_weights[task] is the weight for a given task.
            cre_lines : cre_line information in a batch
            masked_latents : masked latents after passing self attention layer
            original_latents : latents after passing self attention layer
        """
        # print('key of output values is', output_values.keys())
        # print('output values is', output_values['DRIFTING_GRATINGS'].shape) # CRE를 어떻게 여기다가 집어넣을 것인가??
        # {'DRIFTING_GRATINGS': tensor([4, 4, 2, 2, 6, 1, 3, 6, 3, 7, 0, 2, 1, 1, 2, 0, 1, 3, 4, 1, 3, 7, 0, 4,
        # 1, 5, 5, 2, 5, 5, 1, 1, 7, 7, 1, 1, 5, 3, 7, 3, 1, 1, 5, 0, 1, 2, 1, 0,
        # 2, 0, 2, 4, 7, 2, 4, 1, 0, 4, 7, 0, 3, 4, 4, 6, 3, 5, 0, 6, 6, 0, 2, 2,
        # 2, 6, 5, 2, 6, 2, 1, 7, 7, 4, 6, 3, 5, 7, 1, 2, 6, 1, 2, 0, 1, 0, 7, 4,
        # 7, 5, 4, 3, 0, 3, 7, 3, 6, 1, 7, 0, 6, 1, 7, 4, 1, 1, 3, 0, 3, 1, 0, 0,
        # 3, 7, 4, 4, 1, 1, 0, 0], device='cuda:0')} -> decoding ground truth
        
        if self.use_cre_loss:
            ## cre-line
            max_values, _ = torch.max(cre_lines, dim=1)
            # 모든 값이 0인 행 찾기 (모든 값이 0이면 최댓값이 0)
            all_zeros = (max_values == 0)
            # 모든 값이 0인 행은 0으로, 그 외에는 최댓값으로 채우기
            unique_cre_lines = torch.where(all_zeros, torch.zeros_like(max_values), max_values) # shape is (128,)
            output_values['CRE_LINE'] = unique_cre_lines
        
        if self.task in ['movie_decoding_one', 'movie_decoding_three', 'scene_decoding']:
            perceptual_loss = AlexNetPerceptualLoss()
            ssim_loss = SSIMLoss()
            focal_loss = FocalLoss()
            fft_loss = FFTLoss(loss_type='l1')

        if output_batch_index is not None:
            # Inputs were chained, make sure input dimensions make sense
            assert output_latents.dim() == 2
            assert output_decoder_index.dim() == 1
            assert output_batch_index.dim() == 1
            batch_size = output_batch_index.max().item() + 1
        else:
            # Inputs were not chained, make sure input dimensions make sense
            assert output_latents.dim() == 3
            assert output_decoder_index.dim() == 2
            batch_size = output_latents.shape[0]
        
        loss = torch.tensor(0, device=output_latents.device, dtype=torch.float32)

        
        #else:
        taskwise_loss = {}

        if not self.no_decoding:
            if self.task in ['static_gratings', 'drifting_gratings']:
                outputs = [{} for _ in range(batch_size)]
                for decoder_id, spec in self.decoder_specs.items():
                    # the taskid is a universal unique identifier for the task
                    decoder_index = Decoder.from_string(decoder_id).value
                    # get the mask of tokens that belong to this task
                    mask = output_decoder_index == decoder_index
                    if decoder_id == 'CRE_LINE':
                        # CRE에 일일이 디코더 달지 말라고...
                        mask = ~mask
                    # apply the projection
                    task_output = self.projections[decoder_id](output_latents[mask])
                    
                    if not torch.any(mask):
                        # there is not a single token in the batch for this task, so we skip
                        continue
                    # we need to distribute the outputs to their respective samples
                    if self.batch_type == "stacked":
                        token_batch = torch.where(mask)[0]
                    elif self.batch_type == "chained":
                        token_batch = output_batch_index[mask]
                    else:
                        raise ValueError(f"Unknown batch_type: {self.batch_type}")

                    unique_batch_indices = torch.unique(token_batch)
                    for batch_idx in unique_batch_indices:
                        outputs[batch_idx][decoder_id] = task_output[token_batch == batch_idx]
                    # compute loss
                    if output_values is not None:
                        target = output_values[decoder_id]

                        weights = 1.0
                        if (
                            decoder_id in output_weights
                            and output_weights[decoder_id] is not None
                        ):
                            weights = output_weights[decoder_id]

                        taskwise_loss[decoder_id] = compute_loss_or_metric(
                            spec.loss_fn, spec.type, task_output, target, weights, decoder_id
                        )

                    # we need to distribute the outputs to their respective samples
                    if output_batch_index is None:
                        batch_index_filtered_by_decoder = torch.where(mask)[0]
                    else:
                        # Inputs where chained, and we have batch-indices for each token
                        batch_index_filtered_by_decoder = output_batch_index[mask]

                    targeted_batch_elements, batch_index_filtered_by_decoder = torch.unique(
                        batch_index_filtered_by_decoder, return_inverse=True
                    )
                    for i in range(len(targeted_batch_elements)):
                        outputs[targeted_batch_elements[i]][decoder_id] = task_output[
                            batch_index_filtered_by_decoder == i
                        ]

                    if output_values is not None:
                        # Since we calculate a mean across all elements, scale by the number of
                        # items in the batch so we don't get wild swings in loss depending on
                        # whether we have large or small numbers of non-dominant classes.
                        loss = loss + taskwise_loss[decoder_id] * len(targeted_batch_elements)
            else: 
                movie_frames = movie_frames.unsqueeze(dim=1)
                # print('movie frames in multitask readout', movie_frames.shape) # (batch, 1, 32, 64)
                #print('output latents', output_latents.shape) # (batch, 1, 64)
                

                output_tensor = []
                # Neural latents의 분포 확인
                print("Neural latents 통계:", 
                    "평균:", output_latents.mean().item(), 
                    "분산:", output_latents.var().item(),
                    "최소:", output_latents.min().item(),
                    "최대:", output_latents.max().item())
                
                if self.unet_type in ['half_unet', 'mlp']:
                    # print('movie frames', movie_frames.shape) # (batch, 1, 32, 64)
                    for frame in range(movie_frames.shape[1]):
                        # Extract the frame latent vector and expand it
                        # print('output latents', output_latents.shape) # (batch, 1, 64)
                        frame_latent = output_latents[:, frame, :]  # Shape (batch, 64)
                        if 'movie_decoding' in self.task:
                            decoded_frame = self.projections['NATURAL_MOVIE_RECONSTRUCTION'](frame_latent) # (batch, 32, 64)
                        elif self.task == 'scene_decoding':
                            decoded_frame = self.projections['NATURAL_SCENE_RECONSTRUCTION'](frame_latent) # (batch, 32, 64)
                        # Add the decoded frame to the output list
                        output_tensor.append(decoded_frame)
                else:
                    # noise 먹이는 UNet
                    ## frame이 1개 밖에 없음.
                    for frame in range(movie_frames.shape[1]):
                        # Extract the frame latent vector and expand it
                        # print('output latents', output_latents.shape) # (batch, 1, 64)
                        frame_latent = output_latents[:, frame, :]  # Shape (batch, 64)
                        # Apply the decoder to this frame's latent vector
                        noise = torch.randn(frame_latent.size(0), 1, 32, 64).to(output_latents.device) # (batch, 1, 32, 64)
                        if 'movie_decoding' in self.task:
                            decoded_frame = self.projections['NATURAL_MOVIE_RECONSTRUCTION'](noise, frame_latent)
                        elif self.task == 'scene_decoding':
                            decoded_frame = self.projections['NATURAL_SCENE_RECONSTRUCTION'](noise, frame_latent)
                        # print('decoded frame', decoded_frame.shape) # (batch, 32, 64)
                        
                        # Add the decoded frame to the output list
                        output_tensor.append(decoded_frame)

                # Stack decoded frames
                recon_imgs = torch.sigmoid(torch.stack(output_tensor, dim=1))*255 # ([batch, 1, 32, 64])


                print("출력 이미지 값 범위:", recon_imgs.min().item(), recon_imgs.max().item())
                print("원래 이미지 값 범위:", movie_frames.min().item(), movie_frames.max().item())
                
                def expand_to_rgb(tensor):
                    return tensor.expand(-1, 3, -1, -1)
                
                recon_imgs_rgb = expand_to_rgb(recon_imgs)  # [batch, channels, 64, 128]
                movie_rgb = expand_to_rgb(movie_frames).float()  # [batch, 3, 64, 128]

                ### L1 variants ###
                taskwise_loss['L1_loss'] = F.l1_loss(recon_imgs_rgb, movie_rgb) / (255*128) * 20000 * 0.6
                taskwise_loss['focal_loss'] = ssim_loss(recon_imgs_rgb, movie_rgb) / 255*128 * 100 * 0.4
                taskwise_loss['perceptual_loss'] = perceptual_loss(recon_imgs_rgb, movie_rgb) / (255*128*1000)
                taskwise_loss['ssim_loss'] = ssim_loss(recon_imgs_rgb, movie_rgb) / 255*128 * 0.1

                taskwise_loss['fft_loss'] = fft_loss(recon_imgs_rgb.float(), movie_rgb.float()) / (255*128) * 40000 * 0.005

                loss = loss + taskwise_loss['perceptual_loss'] + taskwise_loss['focal_loss'] + taskwise_loss['L1_loss'] + taskwise_loss['ssim_loss']
                outputs = [recon_imgs_rgb, movie_rgb]
            
            if self.use_random_masking_loss:
                masking_loss_weight = 1
                B, N_latents, dim = masked_latents.size()
                masking_num = int(N_latents * self.masking_rate)
                taskwise_loss['random_masking_loss'] = masking_loss_weight * F.mse_loss(masked_latents, original_latents)
                # taskwise_loss['random_masking_loss'] = masking_loss_weight * F.mse_loss(masked_latents[:, -masking_num:, :], original_latents[:, -masking_num:, :])
                loss = loss + taskwise_loss['random_masking_loss'] * len(targeted_batch_elements) * 0.1 # stella added *0.1 for CE weight exp. 250923
                

        else:
            outputs=None
            if self.use_random_masking_loss:
                masking_loss_weight = 10 # min(0.1, current_step / 10000) 
                B, N_latents, dim = masked_latents.size()
                masking_num = int(N_latents * self.masking_rate)
                taskwise_loss['random_masking_loss'] = masking_loss_weight * F.mse_loss(masked_latents[:, -masking_num:, :], original_latents[:, -masking_num:, :])
                loss += taskwise_loss['random_masking_loss'] #* len(targeted_batch_elements)
            elif self.use_cre_loss:
                raise NotImplementedError
        
        if output_values is None:
            return outputs, None, None


        return outputs, loss, taskwise_loss

def prepare_for_multitask_readout(
    data,
    decoder_registry: Dict[str, DecoderSpec],
):
    decoder_index = list()
    timestamps = list()
    values = dict()
    # task_index = dict()
    subtask_index = dict()
    weights = dict()

    config = data.config["multitask_readout"]

    for decoder in config:
        key = decoder["decoder_id"]
        weight = decoder.get("weight", 1.0)
        subtask_weights = decoder.get("subtask_weights", {})

        decoder = decoder_registry[key].__dict__ | decoder  # config overrides registry

        decoder_index.append(Decoder.from_string(key).value)
        values[key] = data.get_nested_attribute(decoder["value_key"])

        # z-scale the values if mean/std are specified in the config file
        if "normalize_mean" in decoder:
            # if mean is a list, its a per-channel mean (usually for x,y coordinates)
            if isinstance(decoder["normalize_mean"], list):
                mean = np.array(decoder["normalize_mean"])
            else:
                mean = decoder["normalize_mean"]
            values[key] = values[key] - mean
        if "normalize_std" in decoder:
            # if std is a list, its a per-channel std (usually for x,y coordinates)
            if isinstance(decoder["normalize_std"], list):
                std = np.array(decoder["normalize_std"])
            else:
                std = decoder["normalize_std"]
            values[key] = values[key] / std

        timestamps.append(data.get_nested_attribute(decoder["timestamp_key"]))

        # here we assume that we won't be running a model at float64 precision
        # TODO do this in decoder spec?
        if values[key].dtype == np.float64:
            values[key] = values[key].astype(np.float32)

        # if decoder["task_index"] is not None:
        #     task_index[key] = data.get_nested_attribute(decoder["task_index"])
        # else:
        #     task_index[key] = np.zeros(len(values[key]), dtype=np.int64)

        if decoder["subtask_key"] is not None:
            subtask_index[key] = data.get_nested_attribute(decoder["subtask_key"])
            num_subtasks = Task.from_string(list(subtask_weights.keys())[0]).max_value()
            subtask_weight_map = np.ones(num_subtasks, dtype=np.float32)
            for subtask, subtask_weight in subtask_weights.items():
                subtask_weight_map[Task.from_string(subtask).value] = subtask_weight

            subtask_weight_map *= weight
            weights[key] = subtask_weight_map[subtask_index[key]]
        else:
            subtask_index[key] = np.zeros(len(values[key]), dtype=np.int64)
            weights[key] = np.ones(len(values[key]), dtype=np.float32) * weight

    # chain
    timestamps, batch = collate(
        [
            (chain(timestamps[i]), track_batch(timestamps[i]))
            for i in range(len(timestamps))
        ]
    )
    decoder_index = torch.tensor(decoder_index)[batch]

    return timestamps, decoder_index, values, weights, subtask_index
