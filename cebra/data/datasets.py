#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Pre-defined datasets."""

import types
from typing import List, Literal, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import torch

import cebra
import cebra.data as cebra_data
import cebra.data.masking as cebra_data_masking
import cebra.helper as cebra_helper
import cebra.io as cebra_io
from cebra.data.datatypes import Batch
from cebra.data.datatypes import BatchIndex
from cebra.data.datatypes import Offset

# Import kirby dataset classes
try:
    from kirby.data import Dataset as KirbyDataset
    from kirby.data import Data as KirbyData
    KIRBY_AVAILABLE = True
except ImportError:
    KIRBY_AVAILABLE = False
    KirbyDataset = None
    KirbyData = None

if TYPE_CHECKING:
    from cebra.models import Model


class TensorDataset(cebra_data.SingleSessionDataset):
    """Discrete and/or continuously indexed dataset based on torch/numpy arrays.

    If dealing with datasets sufficiently small to fit :py:func:`numpy.array` or :py:class:`torch.Tensor`, this
    dataset is sufficient---the sampling auxiliary variable should be specified with a dataloader.
    Based on whether `continuous` and/or `discrete` auxiliary variables are provided, this class
    can be used with the discrete, continuous and/or mixed data loader classes.

    Args:
        neural:
            Array of dtype ``float`` or float Tensor of shape ``(N, D)``, containing neural activity over time.
        continuous:
            Array of dtype ```float`` or float Tensor of shape ``(N, d)``, containing the continuous behavior
            variables over the same time dimension.
        discrete:
            Array of dtype ```int64`` or integer Tensor of shape ``(N, d)``, containing the discrete behavior
            variables over the same time dimension.

    Example:

        >>> import cebra.data
        >>> import torch
        >>> data = torch.randn((100, 30))
        >>> index1 = torch.randn((100, 2))
        >>> index2 = torch.randint(0,5,(100, ))
        >>> dataset = cebra.data.datasets.TensorDataset(data, continuous=index1, discrete=index2)

    """

    def __init__(self,
                 neural: Union[torch.Tensor, npt.NDArray],
                 continuous: Union[torch.Tensor, npt.NDArray] = None,
                 discrete: Union[torch.Tensor, npt.NDArray] = None,
                 offset: Offset = Offset(0, 1),
                 device: str = "cpu"):
        super().__init__(device=device)
        self.neural = self._to_tensor(neural, check_dtype="float").float()
        self.continuous = self._to_tensor(continuous, check_dtype="float")
        self.discrete = self._to_tensor(discrete, check_dtype="int")
        if self.continuous is None and self.discrete is None:
            raise ValueError(
                "You have to pass at least one of the arguments 'continuous' or 'discrete'."
            )
        self.offset = offset

    def _to_tensor(
            self,
            array: Union[torch.Tensor, npt.NDArray],
            check_dtype: Optional[Literal["int",
                                          "float"]] = None) -> torch.Tensor:
        """Convert :py:func:`numpy.array` to :py:class:`torch.Tensor` if necessary and check the dtype.

        Args:
            array: Array to check.
            check_dtype: If not `None`, list of dtypes to which the values in `array`
                must belong to. Defaults to None.

        Returns:
            The `array` as a :py:class:`torch.Tensor`.
        """
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        if check_dtype is not None:
            if check_dtype not in ["int", "float"]:
                raise ValueError(
                    f"check_dtype must be 'int' or 'float', got {check_dtype}")
            if (check_dtype == "int" and not cebra_helper._is_integer(array)
               ) or (check_dtype == "float" and
                     not cebra_helper._is_floating(array)):
                raise TypeError(
                    f"Array has type {array.dtype} instead of {check_dtype}.")
        if cebra_helper._is_floating(array):
            array = array.float()
        if cebra_helper._is_integer(array):
            # NOTE(stes): Required for standardizing number format on
            # windows machines.
            array = array.long()
        return array

    @property
    def input_dimension(self) -> int:
        return self.neural.shape[1]

    @property
    def continuous_index(self):
        if self.continuous is None:
            raise NotImplementedError()
        return self.continuous

    @property
    def discrete_index(self):
        if self.discrete is None:
            raise NotImplementedError()
        return self.discrete

    def __len__(self):
        return len(self.neural)

    def __getitem__(self, index):
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)


class KirbyDatasetAdapter(cebra_data.SingleSessionDataset):
    """Adapter to use kirby.data.Dataset with CEBRA's data loading infrastructure.

    This class wraps a kirby Dataset and makes it compatible with CEBRA's
    SingleSessionDataset interface. It extracts neural data and auxiliary
    variables from kirby Data objects and presents them in CEBRA's format.

    Args:
        kirby_dataset: A kirby.data.Dataset instance containing session data
        session_id: The session ID to extract from the kirby dataset
        continuous_keys: List of keys in kirby Data objects to use as continuous indices
        discrete_keys: List of keys in kirby Data objects to use as discrete indices
        device: Device to store the data on (default: "cpu")

    Example:
        >>> from kirby.data import Dataset as KirbyDataset
        >>> kirby_ds = KirbyDataset(root="data", split="train", include=[...])
        >>> session_id = kirby_ds.session_ids[0]
        >>> adapter = KirbyDatasetAdapter(
        ...     kirby_dataset=kirby_ds,
        ...     session_id=session_id,
        ...     continuous_keys=['timestamps'],
        ... )
    """

    def __init__(
        self,
        kirby_dataset: "KirbyDataset",
        session_id: str,
        continuous_keys: Optional[List[str]] = None,
        discrete_keys: Optional[List[str]] = None,
        neural_key: str = "patches",
        device: str = "cpu",
    ):
        if not KIRBY_AVAILABLE:
            raise ImportError(
                "kirby package is not available. Please ensure kirby is installed "
                "and accessible in your Python environment."
            )

        super().__init__(device=device)
        self.kirby_dataset = kirby_dataset
        self.session_id = session_id
        self.continuous_keys = continuous_keys or []
        self.discrete_keys = discrete_keys or []
        self.neural_key = neural_key
        self.offset = Offset(0, 1)

        # Get the full session data
        self.session_data = kirby_dataset.get_session_data(session_id)

        # Extract neural data
        if hasattr(self.session_data, neural_key):
            neural_data = getattr(self.session_data, neural_key)
            if hasattr(neural_data, 'obj'):
                neural_data = neural_data.obj
            self.neural = self._to_tensor(neural_data).float()
        else:
            raise ValueError(f"Session data does not contain neural key '{neural_key}'")

        # Extract continuous indices
        if self.continuous_keys:
            continuous_list = []
            for key in self.continuous_keys:
                if hasattr(self.session_data, key):
                    data = getattr(self.session_data, key)
                    if hasattr(data, 'obj'):
                        data = data.obj
                    continuous_list.append(self._to_tensor(data))
            if continuous_list:
                self.continuous = torch.cat([
                    d.reshape(-1, 1) if d.ndim == 1 else d
                    for d in continuous_list
                ], dim=1).float()
            else:
                self.continuous = None
        else:
            self.continuous = None

        # Extract discrete indices
        if self.discrete_keys:
            discrete_list = []
            for key in self.discrete_keys:
                if hasattr(self.session_data, key):
                    data = getattr(self.session_data, key)
                    if hasattr(data, 'obj'):
                        data = data.obj
                    discrete_list.append(self._to_tensor(data))
            if discrete_list:
                self.discrete = torch.cat([
                    d.reshape(-1, 1) if d.ndim == 1 else d
                    for d in discrete_list
                ], dim=1).long()
            else:
                self.discrete = None
        else:
            self.discrete = None

    def _to_tensor(self, array: Union[torch.Tensor, npt.NDArray]) -> torch.Tensor:
        """Convert numpy array to torch tensor if necessary."""
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        return array

    @property
    def input_dimension(self) -> int:
        return self.neural.shape[1]

    @property
    def continuous_index(self):
        if self.continuous is None:
            raise NotImplementedError()
        return self.continuous

    @property
    def discrete_index(self):
        if self.discrete is None:
            raise NotImplementedError()
        return self.discrete

    def __len__(self):
        return len(self.neural)

    def __getitem__(self, index):
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)


def _assert_datasets_same_device(
        datasets: List[cebra_data.SingleSessionDataset]) -> str:
    """Checks if the list of datasets are all on the same device.

    Args:
        datasets: List of datasets.

    Returns:
        The device name if all datasets are on the same device.

    Raises:
        ValueError: If datasets are not all on the same device.
    """
    devices = set([dataset.device for dataset in datasets])
    if len(devices) != 1:
        raise ValueError("Datasets are not all on the same device")
    return devices.pop()


class DatasetCollection(cebra_data.MultiSessionDataset):
    """Multi session dataset made up of a list of datasets.

    Args:
        *datasets: Collection of datasets to add to the collection. The order
            will be maintained for indexing.

    Example:

        >>> import cebra.data
        >>> import torch
        >>> session1 = torch.randn((100, 30))
        >>> session2 = torch.randn((100, 50))
        >>> index1 = torch.randn((100, 4))
        >>> index2 = torch.randn((100, 4)) # same index dim as index1
        >>> dataset = cebra.data.DatasetCollection(
        ...               cebra.data.TensorDataset(session1, continuous=index1),
        ...               cebra.data.TensorDataset(session2, continuous=index2))

    """

    def _has_not_none_attribute(self, obj, key):
        """Check if obj.key exists.

        Returns:
            ``True`` if the key exists and is not ``None``. ``False`` if
            either the key does not exist, the attribute has value ``None``
            or the access raises a ``NotImplementedError``.
        """
        try:
            if hasattr(obj, key):
                if getattr(obj, key) is not None:
                    return True
        except NotImplementedError:
            return False
        return False

    def _unpack_dataset_arguments(
        self, datasets: Tuple[cebra_data.SingleSessionDataset]
    ) -> List[cebra_data.SingleSessionDataset]:
        if len(datasets) == 0:
            raise ValueError("Need to supply at least one dataset.")
        elif len(datasets) == 1:
            (dataset_generator,) = datasets
            if isinstance(dataset_generator, types.GeneratorType):
                return list(dataset_generator)
            else:
                raise ValueError(
                    "You need to specify either a single generator, "
                    "or multiple SingleSessionDataset instances.")
        else:
            return datasets

    def __init__(
        self,
        *datasets: cebra_data.SingleSessionDataset,
    ):
        self._datasets: List[
            cebra_data.SingleSessionDataset] = self._unpack_dataset_arguments(
                datasets)

        device = _assert_datasets_same_device(self._datasets)
        super().__init__(device=device)

        continuous = all(
            self._has_not_none_attribute(session, "continuous_index")
            for session in self.iter_sessions())
        discrete = all(
            self._has_not_none_attribute(session, "discrete_index")
            for session in self.iter_sessions())

        if not (continuous or discrete):
            raise ValueError(
                "The provided datasets need to define either continuous or discrete indices, "
                "or both. Continuous: {continuous}; discrete: {discrete}. "
                "Note that _all_ provided datasets need to define the indexing function of choice."
            )

        if continuous:
            self._cindex = torch.cat(list(
                self._iter_property("continuous_index")),
                                     dim=0)
        else:
            self._cindex = None
        if discrete:
            self._dindex = torch.cat(list(
                self._iter_property("discrete_index")),
                                     dim=0)
        else:
            self._dindex = None

    @property
    def num_sessions(self) -> int:
        """The number of sessions in the dataset."""
        return len(self._datasets)

    @property
    def input_dimension(self):
        return super().input_dimension

    def get_input_dimension(self, session_id: int) -> int:
        """Get the feature dimension of the required session.

        Args:
            session_id: The session ID, an integer between 0 and
                :py:attr:`num_sessions`.

        Returns:
            A single session input dimension for the requested session id.
        """
        return self.get_session(session_id).input_dimension

    def get_session(self, session_id: int) -> cebra_data.SingleSessionDataset:
        """Get the dataset for the specified session.

        Args:
            session_id: The session ID, an integer between 0 and
                :py:attr:`num_sessions`.

        Returns:
            A single session dataset for the requested session
            id.
        """
        return self._datasets[session_id]

    @property
    def continuous_index(self) -> torch.Tensor:
        return self._cindex

    @property
    def discrete_index(self) -> torch.Tensor:
        return self._dindex

    def _apply(self, func):
        return (func(data) for data in self.iter_sessions())

    def _iter_property(self, attr):
        return (getattr(data, attr) for data in self.iter_sessions())

    @classmethod
    def from_kirby_dataset(
        cls,
        kirby_dataset: "KirbyDataset",
        continuous_keys: Optional[List[str]] = None,
        discrete_keys: Optional[List[str]] = None,
        neural_key: str = "patches",
        device: str = "cpu",
        session_ids: Optional[List[str]] = None,
    ) -> "DatasetCollection":
        """Create a DatasetCollection from a kirby Dataset.

        This class method provides a convenient way to convert a kirby Dataset
        containing multiple sessions into a CEBRA DatasetCollection. Each session
        in the kirby Dataset will be wrapped in a KirbyDatasetAdapter and added
        to the collection.

        Args:
            kirby_dataset: A kirby.data.Dataset instance containing multiple sessions
            continuous_keys: List of keys to extract as continuous indices
                (e.g., ['timestamps'])
            discrete_keys: List of keys to extract as discrete indices
                (e.g., ['unit_cre_line'])
            neural_key: Key for neural data in kirby Data objects (default: "patches")
            device: Device to store data on (default: "cpu")
            session_ids: Optional list of specific session IDs to include.
                If None, all sessions will be included.

        Returns:
            A DatasetCollection containing adapted kirby sessions

        Example:
            >>> from kirby.data import Dataset as KirbyDataset
            >>> from cebra.data import DatasetCollection
            >>> kirby_ds = KirbyDataset(root="data", split="train", include=[...])
            >>> collection = DatasetCollection.from_kirby_dataset(
            ...     kirby_dataset=kirby_ds,
            ...     continuous_keys=['timestamps'],
            ...     discrete_keys=['unit_cre_line'],
            ... )
        """
        if not KIRBY_AVAILABLE:
            raise ImportError(
                "kirby package is not available. Please ensure kirby is installed "
                "and accessible in your Python environment."
            )

        # Determine which sessions to include
        if session_ids is None:
            session_ids = kirby_dataset.session_ids

        # Create adapters for each session
        adapters = []
        for session_id in session_ids:
            adapter = KirbyDatasetAdapter(
                kirby_dataset=kirby_dataset,
                session_id=session_id,
                continuous_keys=continuous_keys,
                discrete_keys=discrete_keys,
                neural_key=neural_key,
                device=device,
            )
            adapters.append(adapter)

        # Create and return the DatasetCollection
        return cls(*adapters)


# TODO(stes): This should be a single session dataset?
class DatasetxCEBRA(cebra_io.HasDevice, cebra_data_masking.MaskedMixin):
    """Dataset class for xCEBRA models.

    This class handles neural data and associated labels for xCEBRA models, providing
    functionality for data loading and batch preparation.

    Attributes:
        neural: Neural data as a torch.Tensor or numpy array
        labels: Labels associated with the data
        offset: Offset for the dataset

    Args:
        neural: Neural data as a torch.Tensor or numpy array
        device: Device to store the data on (default: "cpu")
        **labels: Additional keyword arguments for labels associated with the data
    """

    def __init__(
        self,
        neural: Union[torch.Tensor, npt.NDArray],
        device="cpu",
        **labels,
    ):
        super().__init__(device)
        self.neural = neural
        self.labels = labels
        self.offset = Offset(0, 1)

    @property
    def input_dimension(self) -> int:
        """Get the input dimension of the neural data.

        Returns:
            The number of features in the neural data
        """
        return self.neural.shape[1]

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            Number of samples in the dataset
        """
        return len(self.neural)

    def configure_for(self, model: "Model"):
        """Configure the dataset offset for the provided model.

        Call this function before indexing the dataset. This sets the
        :py:attr:`offset` attribute of the dataset.

        Args:
            model: The model to configure the dataset for.
        """
        self.offset = model.get_offset()

    def expand_index(self, index: torch.Tensor) -> torch.Tensor:
        """Expand indices based on the configured offset.

        Args:
            index: A one-dimensional tensor of type long containing indices
                to select from the dataset.

        Returns:
            An expanded index of shape ``(len(index), len(self.offset))`` where
            the elements will be
            ``expanded_index[i,j] = index[i] + j - self.offset.left`` for all ``j``
            in ``range(0, len(self.offset))``.

        Note:
            Requires the :py:attr:`offset` to be set.
        """
        offset = torch.arange(-self.offset.left,
                              self.offset.right,
                              device=index.device)

        index = torch.clamp(index, self.offset.left,
                            len(self) - self.offset.right)

        return index[:, None] + offset[None, :]

    def __getitem__(self, index):
        """Get item(s) from the dataset at the specified index.

        Args:
            index: Index or indices to retrieve

        Returns:
            The neural data at the specified indices, with dimensions transposed
        """
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)

    def load_batch_supervised(self, index: Batch,
                              labels_supervised) -> torch.Tensor:
        """Load a batch for supervised learning.

        Args:
            index: Batch indices for reference data
            labels_supervised: Labels to load for supervised learning

        Returns:
            Batch containing reference data and corresponding labels
        """
        assert index.negative is None
        assert index.positive is None
        labels = [
            self.labels[label].to(self.device) for label in labels_supervised
        ]

        return Batch(
            reference=self[index.reference],
            positive=[label[index.reference] for label in labels],
            negative=None,
        )

    def load_batch_contrastive(self, index: BatchIndex) -> Batch:
        """Load a batch for contrastive learning.

        Args:
            index: BatchIndex containing reference, positive and negative indices

        Returns:
            Batch containing reference, positive and negative samples
        """
        assert isinstance(index.positive, list)
        return Batch(
            reference=self[index.reference],
            positive=[self[idx] for idx in index.positive],
            negative=self[index.negative],
        )


class UnifiedDataset(DatasetCollection):
    """Multi session dataset made up of a list of datasets, considered as a unique session.

    Considering the sessions as a unique session, or pseudo-session, is used to later train a single
    model for all the sessions, even if they originally contain a variable number of neurons.
    To do that, we sample ref/pos/neg for each session and concatenate them along the neurons axis.

    For instance, for a batch size ``batch_size``, we sample ``(batch_size, num_neurons(session), offset)`` for
    each type of samples (ref/pos/neg) and then concatenate so that the final :py:class:`cebra.data.datatypes.Batch`
    is of shape ``(batch_size, total_num_neurons, offset)``, with ``total_num_neurons`` is  the sum of all the
    ``num_neurons(session)``.
    """

    def __init__(self, *datasets: cebra_data.SingleSessionDataset):
        super().__init__(*datasets)

    @property
    def input_dimension(self) -> int:
        """Returns the sum of the input dimension for each session."""
        return np.sum([
            self.get_input_dimension(session_id)
            for session_id in range(self.num_sessions)
        ])

    def _get_batches(self, index):
        """Return the data at the specified index location."""
        return [
            cebra_data.Batch(
                reference=self.get_session(session_id)[
                    index.reference[session_id]],
                positive=self.get_session(session_id)[
                    index.positive[session_id]],
                negative=self.get_session(session_id)[
                    index.negative[session_id]],
            ) for session_id in range(self.num_sessions)
        ]

    def configure_for(self, model: "cebra.models.Model"):
        """Configure the dataset offset for the provided model.

        Call this function before indexing the dataset. This sets the
        :py:attr:`~.Dataset.offset` attribute of the dataset.

        Args:
            model: The model to configure the dataset for.
        """
        for i, session in enumerate(self.iter_sessions()):
            session.configure_for(model)

    def load_batch(self, index: BatchIndex) -> Batch:
        """Return the data at the specified index location.

        Concatenate batches for each sessions on the number of neurons axis.

        Args:
            batches: List of :py:class:`cebra.data.datatypes.Batch` sampled for each session. An instance
                :py:class:`cebra.data.datatypes.Batch` of the list is of shape ``(batch_size, num_neurons(session), offset)``.

        Returns:
            A :py:class:`cebra.data.datatypes.Batch`, of shape ``(batch_size, total_num_neurons, offset)``, where
            ``total_num_neurons`` is  the sum of all the ``num_neurons(session)``
        """
        batches = self._get_batches(index)

        if hasattr(self, "apply_mask"):
            # If the dataset has a mask, apply it to the data.
            batch = cebra_data.Batch(
                reference=self.apply_mask(
                    torch.cat([batch.reference for batch in batches], dim=1)),
                positive=self.apply_mask(
                    torch.cat([batch.positive for batch in batches], dim=1)),
                negative=self.apply_mask(
                    torch.cat([batch.negative for batch in batches], dim=1)),
            )
        else:
            batch = cebra_data.Batch(
                reference=torch.cat([batch.reference for batch in batches],
                                    dim=1),
                positive=torch.cat([batch.positive for batch in batches],
                                   dim=1),
                negative=torch.cat([batch.negative for batch in batches],
                                   dim=1),
            )
        return batch

    def __getitem__(self, args) -> List[Batch]:
        """Return a set of samples from all sessions."""

        session_id, index = args
        return self.get_session(session_id).__getitem__(index)
