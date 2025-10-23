"""
Example: Using Kirby Dataset with CEBRA

This example demonstrates how to integrate kirby's Dataset with CEBRA's
data loading infrastructure for both single-session and multi-session training.
"""

import sys
import torch

# Add CEBRA to path if needed
# sys.path.append('/path/to/CEBRA')

from kirby.data import Dataset as KirbyDataset
from cebra.data import DatasetCollection
from cebra.data.datasets import KirbyDatasetAdapter


def example_single_session():
    """Example: Using a single session from kirby Dataset with CEBRA."""
    print("=" * 60)
    print("Example 1: Single Session")
    print("=" * 60)

    # Create kirby Dataset
    kirby_ds = KirbyDataset(
        root="data",
        split="train",
        include=[{
            "selection": [{
                "dandiset": "allen_brain_observatory_calcium",
            }]
        }],
        task='movie_decoding_one',
    )

    print(f"Total sessions in kirby dataset: {len(kirby_ds.session_ids)}")

    # Get first session
    session_id = kirby_ds.session_ids[0]
    print(f"Selected session: {session_id}")

    # Convert to CEBRA format
    cebra_dataset = KirbyDatasetAdapter(
        kirby_dataset=kirby_ds,
        session_id=session_id,
        continuous_keys=['timestamps'],
        discrete_keys=['unit_cre_line'],
        neural_key='patches',
        device='cpu',  # Change to 'cuda' if GPU is available
    )

    print(f"Dataset length: {len(cebra_dataset)}")
    print(f"Input dimension: {cebra_dataset.input_dimension}")
    print(f"Continuous index shape: {cebra_dataset.continuous_index.shape}")
    print(f"Discrete index shape: {cebra_dataset.discrete_index.shape}")

    # Example: Get a sample
    sample_index = torch.tensor([0, 1, 2, 3, 4])
    sample = cebra_dataset[sample_index]
    print(f"Sample shape: {sample.shape}")

    print("\n")


def example_multi_session():
    """Example: Using multiple sessions with DatasetCollection."""
    print("=" * 60)
    print("Example 2: Multi-Session Collection")
    print("=" * 60)

    # Create kirby Dataset with multiple sessions
    kirby_ds = KirbyDataset(
        root="data",
        split="train",
        include=[{
            "selection": [{
                "dandiset": "allen_brain_observatory_calcium",
            }]
        }],
        task='movie_decoding_one',
    )

    print(f"Total sessions: {len(kirby_ds.session_ids)}")

    # Convert to CEBRA DatasetCollection
    collection = DatasetCollection.from_kirby_dataset(
        kirby_dataset=kirby_ds,
        continuous_keys=['timestamps'],
        discrete_keys=['unit_cre_line'],
        neural_key='patches',
        device='cpu',
    )

    print(f"Number of sessions in collection: {collection.num_sessions}")

    # Access individual sessions
    for i in range(min(3, collection.num_sessions)):
        session = collection.get_session(i)
        print(f"Session {i}:")
        print(f"  - Length: {len(session)}")
        print(f"  - Input dimension: {session.input_dimension}")

    print("\n")


def example_selected_sessions():
    """Example: Using only specific sessions."""
    print("=" * 60)
    print("Example 3: Selected Sessions Only")
    print("=" * 60)

    # Create kirby Dataset
    kirby_ds = KirbyDataset(
        root="data",
        split="train",
        include=[{
            "selection": [{
                "dandiset": "allen_brain_observatory_calcium",
            }]
        }],
        task='movie_decoding_one',
    )

    # Select specific sessions
    selected_session_ids = kirby_ds.session_ids[:3]  # First 3 sessions
    print(f"Selected sessions: {selected_session_ids}")

    # Convert only selected sessions
    collection = DatasetCollection.from_kirby_dataset(
        kirby_dataset=kirby_ds,
        continuous_keys=['timestamps'],
        session_ids=selected_session_ids,
        device='cpu',
    )

    print(f"Number of sessions in collection: {collection.num_sessions}")

    print("\n")


def example_inspect_kirby_data():
    """Example: Inspecting available keys in kirby Data objects."""
    print("=" * 60)
    print("Example 4: Inspecting Kirby Data")
    print("=" * 60)

    # Create kirby Dataset
    kirby_ds = KirbyDataset(
        root="data",
        split="train",
        include=[{
            "selection": [{
                "dandiset": "allen_brain_observatory_calcium",
            }]
        }],
        task='movie_decoding_one',
    )

    # Get first session data
    session_id = kirby_ds.session_ids[0]
    session_data = kirby_ds.get_session_data(session_id)

    print(f"Session: {session_id}")
    print(f"Available keys: {session_data.keys}")

    # Inspect each key
    for key in session_data.keys[:10]:  # First 10 keys
        data = getattr(session_data, key)
        if hasattr(data, 'obj'):
            data = data.obj
        if hasattr(data, 'shape'):
            print(f"  - {key}: shape={data.shape}, dtype={data.dtype}")
        else:
            print(f"  - {key}: {type(data)}")

    print("\n")


def example_with_cebra_training():
    """Example: Using kirby Dataset with CEBRA training."""
    print("=" * 60)
    print("Example 5: Training with CEBRA")
    print("=" * 60)

    try:
        from cebra.data.single_session import ContinuousDataLoader
        import cebra

        # Create kirby Dataset
        kirby_ds = KirbyDataset(
            root="data",
            split="train",
            include=[{
                "selection": [{
                    "dandiset": "allen_brain_observatory_calcium",
                }]
            }],
            task='movie_decoding_one',
        )

        # Convert to CEBRA format
        session_id = kirby_ds.session_ids[0]
        cebra_dataset = KirbyDatasetAdapter(
            kirby_dataset=kirby_ds,
            session_id=session_id,
            continuous_keys=['timestamps'],
            neural_key='patches',
            device='cpu',
        )

        # Create data loader
        loader = ContinuousDataLoader(
            dataset=cebra_dataset,
            num_steps=10,
            batch_size=512,
        )

        print(f"DataLoader created successfully")
        print(f"  - Dataset length: {len(cebra_dataset)}")
        print(f"  - Input dimension: {cebra_dataset.input_dimension}")

        # Get a batch (this will actually sample from the dataset)
        batch_idx = loader.get_indices(num_samples=512)
        print(f"  - Sample batch index keys: {batch_idx._fields}")

        print("\nReady for CEBRA training!")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure CEBRA is properly installed and configured.")

    print("\n")


if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("Kirby Dataset Integration with CEBRA - Examples")
    print("=" * 60)
    print("\n")

    # Note: Uncomment the examples you want to run
    # Make sure to adjust paths and parameters according to your setup

    print("To run these examples:")
    print("1. Ensure kirby Dataset is properly set up with data")
    print("2. Adjust the 'root' path in the examples")
    print("3. Uncomment the example functions below")
    print("\n")

    # example_single_session()
    # example_multi_session()
    # example_selected_sessions()
    # example_inspect_kirby_data()
    # example_with_cebra_training()

    print("Examples completed!")
