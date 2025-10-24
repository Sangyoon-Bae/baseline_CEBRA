import math
import logging
from typing import List, Dict, Tuple, Optional
from functools import cached_property

import torch

from kirby.data.dataset import DatasetIndex


class RandomFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the
    :obj:`interval_dict` parameter. :obj:`interval_dict` is a dictionary where the keys
    are the session ids and the values are lists of tuples representing the
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:

    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        interval_dict (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        interval_dict: Dict[str, List[Tuple[float, float]]],
        window_length: float,
        generator: Optional[torch.Generator] = None, # 기본값 None 추가
        drop_short: bool = True,
    ):
        self.interval_dict = interval_dict
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.interval_dict.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} ({interval_length:.3f}s) is too short to sample from. "
                            f"Minimum length is {self.window_length}s."
                        )

                # 이 부분은 원래 코드와 동일하게 유지 (샘플 개수 추정)
                # 참고: __iter__의 로직과 약간 다를 수 있음 (특히 random offset과 마지막 샘플 추가 로직 때문에)
                # 정확한 길이를 원한다면 __iter__와 동일한 로직으로 계산하거나, __iter__에서 생성된 인덱스 개수를 세야 함
                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped:.2f} seconds of data due to short "
                f"intervals. Estimated remaining samples: {num_samples} "
                f"(approx. {num_samples * self.window_length:.2f} seconds)."
            )

        # 중요: num_samples가 0일 경우 _estimated_len에서 에러를 내기보다 __len__이나 __iter__에서 처리하는 것이 일반적
        # if num_samples == 0 and not any(total_short_dropped > 0 for ...): # 모든 interval이 0.03 미만인 경우 등 고려
        #     raise ValueError("No valid intervals found to sample from.")

        return num_samples

    def __len__(self):
        # _estimated_len 계산 시 에러가 발생할 수 있으므로 try-except 고려 가능
        length = self._estimated_len
        # if length == 0:
        #     # 여기서 경고나 에러를 발생시킬 수도 있음
        #     print("Warning: Estimated length of sampler is 0.")
        return length

    def __iter__(self):
        all_indices = [] # 모든 인덱스를 저장할 리스트

        for session_name, sampling_intervals in self.interval_dict.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                
                if interval_length < self.window_length:
                    #print('session name', session_name, 'interval length', interval_length)
                    if self.drop_short:
                        continue # 다음 interval로
                    else:
                        # _estimated_len과 동일한 에러 메시지
                        raise ValueError(
                           f"Interval {(start, end)} ({interval_length:.3f}s) is too short to sample from. "
                           f"Minimum length is {self.window_length}s."
                        )

                # --- 각 유효한 interval에 대해 인덱스 생성 ---
                # sample a random offset
                left_offset = (
                    torch.rand(1, generator=self.generator).item() * self.window_length
                )

                # arange로 생성되는 시작점들
                potential_starts = torch.arange(
                    start + left_offset,
                    end, # end는 포함하지 않음
                    self.window_length,
                    dtype=torch.float64,
                )

                # 각 시작점에 대해 (start, end) 인덱스 생성
                # 마지막 window가 interval의 끝(end)을 넘지 않도록 확인
                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in potential_starts
                    if (t + self.window_length) <= end # 이 조건이 중요!
                ]

                # 생성된 인덱스들을 전체 리스트에 추가
                if len(indices_) > 0:
                    all_indices.extend(indices_)
                    # 마지막 인덱스의 끝 시간 기준으로 right_offset 계산
                    # 이 right_offset은 원래 코드의 추가 샘플 로직에 사용됨
                    right_offset_from_last_sample = end - all_indices[-1].end
                else:
                    # arange로 샘플이 하나도 안 나온 경우
                    right_offset_from_last_sample = interval_length - left_offset # 남은 전체 길이


                # --- 원래 코드의 추가 샘플 로직 (필요에 따라 유지 또는 제거/수정) ---
                # 설명: 랜덤 오프셋 때문에 버려지는 앞/뒤 조각(left_offset, right_offset)의 합이
                # window_length보다 크거나 같으면, 그 공간에서 샘플 하나를 더 뽑겠다는 의도로 보임.
                # 이는 __len__의 floor 계산과 약간의 불일치를 만들 수 있지만, 데이터를 최대한 활용하려는 시도.
                
                if right_offset_from_last_sample + left_offset >= self.window_length:
                    # 더 긴 쪽의 경계에 맞춰서 샘플 추가 (원래 코드 로직)
                    if right_offset_from_last_sample > left_offset:
                        # 구간 끝에 맞춰서 추가
                        all_indices.append(
                            DatasetIndex(session_name, end - self.window_length, end)
                        )
                    else:
                        # 구간 시작에 맞춰서 추가
                        all_indices.append(
                            DatasetIndex(
                                session_name, start, start + self.window_length
                            )
                        )
                # -----------------------------------------------------------

        # 모든 interval 처리가 끝난 후
        if not all_indices:
             # len(self) > 0 임에도 인덱스가 생성되지 않은 경우 (로직 오류 가능성)
             # 또는 len(self) == 0 인 경우 (이 경우는 정상)
             if len(self) > 0:
                 # 이 경우는 뭔가 잘못된 것. _estimated_len과 __iter__ 로직 불일치 가능성.
                 # 예를 들어 추가 샘플 로직 등에서 미묘한 차이가 발생할 수 있음.
                 logging.error(f"Sampler estimated length {len(self)}, but generated 0 indices. Check logic.")
                 # 빈 iterator 반환 (오류 대신)
                 # yield from [] # Python 3.3+
                 return # 빈 iterator 반환
             else:
                 # 추정 길이가 0이었고 실제로도 인덱스가 없으면 정상 종료
                 # raise ValueError("All intervals are too short or invalid to sample from.") # 여기서 에러 발생 가능
                 return # 빈 iterator 반환

        print('length of all indices', len(all_indices))
        # --- 모든 인덱스를 모은 후, 한 번만 셔플링 ---
        indices_to_yield = torch.randperm(len(all_indices), generator=self.generator)
        for i in indices_to_yield:
            yield all_indices[i]

'''
class RandomFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the
    :obj:`interval_dict` parameter. :obj:`interval_dict` is a dictionary where the keys
    are the session ids and the values are lists of tuples representing the
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:

    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        interval_dict (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        interval_dict: Dict[str, List[Tuple[float, float]]],
        window_length: float,
        generator: Optional[torch.Generator],
        drop_short: bool = True,
    ):
        self.interval_dict = interval_dict
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.interval_dict.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                if interval_length >= 0.03:
                    if interval_length < self.window_length:
                        if self.drop_short:
                            total_short_dropped += interval_length
                            continue
                        else:
                            raise ValueError(
                                f"Interval {(start, end)} is too short to sample from. "
                                f"Minimum length is {self.window_length}."
                            )

                    # print('interval length', interval_length)
                    # print('window length', self.window_length)

                    num_samples += math.floor(interval_length / self.window_length)
                else:
                    print('interval length', interval_length)

                    pass

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        return self._estimated_len

    def __iter__(self):
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")

        indices = []
        for session_name, sampling_intervals in self.interval_dict.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                if interval_length >= 0.03:
                    if interval_length < self.window_length:
                        if self.drop_short:
                            continue
                        else:
                            raise ValueError(
                                f"Interval {(start, end)} is too short to sample from. "
                                f"Minimum length is {self.window_length}."
                            )

                    # sample a random offset
                    left_offset = (
                        torch.rand(1, generator=self.generator).item() * self.window_length
                    )

                    indices_ = [
                        DatasetIndex(
                            session_name, t.item(), (t + self.window_length).item()
                        )
                        for t in torch.arange(
                            start + left_offset,
                            end,
                            self.window_length,
                            dtype=torch.float64,
                        )
                        if t + self.window_length <= end
                    ]

                    if len(indices_) > 0:
                        indices.extend(indices_)
                        right_offset = end - indices[-1].end
                    else:
                        right_offset = end - start - left_offset

                    # if there is one sample worth of data, add it
                    # this ensures that the number of samples is always consistent
                    if right_offset + left_offset >= self.window_length:
                        if right_offset > left_offset:
                            indices.append(
                                DatasetIndex(session_name, end - self.window_length, end)
                            )
                        else:
                            indices.append(
                                DatasetIndex(
                                    session_name, start, start + self.window_length
                                )
                            )
                else:
                    pass
            # shuffle
            for idx in torch.randperm(len(indices), generator=self.generator):
                yield indices[idx]
    def __getitem__(self, index):
        return self._indices[index]
'''

class SequentialFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows sequentially, always in the same order. The
    sampling intervals are defined in the :obj:`interval_dict` parameter.
    :obj:`interval_dict` is a dictionary where the keys are the session ids and the
    values are lists of tuples representing the start and end of the intervals
    from which to sample.

    If the length of a sequence is not evenly divisible by the step, the last
    window will be added with an overlap with the previous window. This is to ensure
    that the entire sequence is covered.

    Args:
        interval_dict (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        step (Optional[float], optional): Step size between windows. If None, it
            defaults to `window_length`. Defaults to None.
    """

    def __init__(
        self,
        *,
        interval_dict: Dict[str, List[Tuple[float, float]]],
        window_length: float,
        step: Optional[float] = None,
        drop_short=False,
    ):
        self.interval_dict = interval_dict
        self.window_length = window_length
        self.step = step or window_length
        self.drop_short = drop_short

        assert self.step > 0, "Step must be greater than 0."
        assert self.step <= self.window_length, "Step must be less than window length."

    # we cache the indices since they are deterministic
    @cached_property
    def _indices(self) -> List[DatasetIndex]:
        indices = []
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.interval_dict.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                if interval_length >= 0.033:
                    if interval_length < self.window_length:
                        if self.drop_short:
                            total_short_dropped += interval_length
                            continue
                        else:
                            raise ValueError(
                                f"Interval {(start, end)} is too short to sample from. "
                                f"Minimum length is {self.window_length}."
                            )

                    indices_ = [
                        DatasetIndex(
                            session_name, t.item(), (t + self.window_length).item()
                        )
                        for t in torch.arange(start, end, self.step, dtype=torch.float64)
                        if t + self.window_length <= end
                    ]
                    # print(f"{session_name}: {len(indices_)} samples")

                    indices.extend(indices_)

                    # we need to make sure that the entire interval is covered
                    if indices_[-1].end < end:
                        indices.append(
                            DatasetIndex(session_name, end - self.window_length, end)
                        )

            if self.drop_short and total_short_dropped > 0:
                num_samples = len(indices)
                logging.warning(
                    f"Skipping {total_short_dropped} seconds of data due to short "
                    f"intervals. Remaining: {num_samples * self.window_length} seconds."
                )
                if num_samples == 0:
                    raise ValueError("All intervals are too short to sample from.")

        return indices

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        yield from self._indices

    def __getitem__(self, index):
        return self._indices[index]


class TrialSampler(torch.utils.data.Sampler):
    r"""Randomly samples a single trial interval from the given intervals.

    Args:
        interval_dict (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        interval_dict: Dict[str, List[Tuple[float, float]]],
        generator: Optional[torch.Generator] = None,
        shuffle: bool = True,
    ):
        self.interval_dict = interval_dict
        self.generator = generator
        self.shuffle = shuffle

    def __len__(self):
        return sum(len(intervals) for intervals in self.interval_dict.values())

    def __iter__(self):
        # Flatten the intervals from all sessions into a single list
        all_intervals = [
            (session_id, start, end)
            for session_id, intervals in self.interval_dict.items()
            for start, end in intervals
        ]

        indices = [
            DatasetIndex(session_id, start, end)
            for session_id, start, end in all_intervals
        ]

        if self.shuffle:
            # Yield a single DatasetIndex representing the selected interval
            for idx in torch.randperm(len(indices), generator=self.generator):
                yield indices[idx]
        else:
            yield from indices


class DistributedSamplerWrapper(torch.utils.data.Sampler):
    r"""Wraps a sampler to be used in a distributed setting. This sampler will
    only return indices that are assigned to the current process based on the
    rank and num_replicas.
    Args:
        sampler (torch.utils.data.Sampler): The original sampler to wrap.
        num_replicas (int): Number of processes participating in the distributed
            training.
        rank (int): Rank of the current process.
    Example:
        >>> sampler = SequentialFixedWindowSampler(interval_dict, window_length=10)
        >>> dist_sampler = DistributedSamplerWrapper(sampler)
        >>> loader = torch.utils.data.DataLoader(dataset, sampler=dist_sampler)
        # Before starting the training loop, set the rank and num_replicas attributes:
        >>> dist_sampler.set_params(trainer.world_size, trainer.global_rank)
        # Now use it
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank

    def set_params(self, num_replicas, rank):
        logging.info(
            f"Setting distributed sampler params: "
            f"num_replicas={num_replicas}, rank={rank}"
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def _check_params(self):
        return (self.num_replicas is not None) and (self.rank is not None)

    def rank_len(self):
        r"""Returns the number of samples assigned to the current process."""
        total_len = len(self.sampler)
        evenly_split = total_len // self.num_replicas
        extra = int((total_len % self.num_replicas) < self.rank)
        return evenly_split + extra

    def __len__(self):
        r"""Returns the number of samples assigned to the current process if
        the rank and num_replicas are set. Otherwise, returns the total number
        of samples in the original sampler.
        """
        if not self._check_params():
            return len(self.sampler)
        else:
            return self.rank_len()

    def __iter__(self):
        assert (
            self._check_params()
        ), "Rank and num_replicas must be set before using the distributed sampler."
        indices = list(self.sampler)
        indices = indices[self.rank : len(indices) : self.num_replicas]
        return iter(indices)