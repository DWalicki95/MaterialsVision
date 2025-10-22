#!/usr/bin/env python3
"""
Script to filter images from train/test directories based on partial name
matches
and copy them to an output directory with the same structure.

Usage:
    python filter_images.py

Example:
    # Interactive mode - the script will ask for input
    python filter_images.py

    # Programmatic usage
    from filter_images import filter_and_copy_images
    filter_and_copy_images(
        train_dir="/path/to/train",
        test_dir="/path/to/test",
        name_patterns=["cell", "nucleus"],
        output_dir="/path/to/output"
    )
"""

import shutil
from pathlib import Path
from typing import List, Set


def filter_and_copy_images(
    train_dir: str,
    test_dir: str,
    name_patterns: List[str],
    output_dir: str,
    image_extensions: Set[str] = None
) -> None:
    """
    Filter images from train/test directories based on partial name matches.

    Parameters
    ----------
    train_dir : str
        Path to the training images directory
    test_dir : str
        Path to the test images directory
    name_patterns : List[str]
        List of strings to match in image names (partial matches)
    output_dir : str
        Path to the output directory where filtered images will be copied
    image_extensions : Set[str], optional
        Set of supported image file extensions, by default
        {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    Returns
    -------
    None
        This function does not return a value.

    Notes
    -----
    The function creates train/ and test/ subdirectories in the output
    directory and copies only the images whose names contain any of the
    specified patterns. Matching is case-insensitive.
    """
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Convert to Path objects
    train_path = Path(train_dir)
    test_path = Path(test_dir)
    output_path = Path(output_dir)

    # Validate input directories
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training directory does not exist: {train_dir}"
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test directory does not exist: {test_dir}"
        )

    # Create output directory structure
    output_train = output_path / "train"
    output_test = output_path / "test"
    output_train.mkdir(parents=True, exist_ok=True)
    output_test.mkdir(parents=True, exist_ok=True)

    print(f"Filtering images with patterns: {name_patterns}")
    print("Input directories:")
    print(f"   Train: {train_dir}")
    print(f"   Test:  {test_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Process training images
    train_stats = _process_directory(
        train_path, output_train, name_patterns, image_extensions, "Train"
    )

    # Process test images
    test_stats = _process_directory(
        test_path, output_test, name_patterns, image_extensions, "Test"
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Train directory:")
    print(f"   Total images found: {train_stats['total']}")
    print(f"   Images matching patterns: {train_stats['matched']}")
    print(f"   Images copied: {train_stats['copied']}")
    print(f"   Errors: {train_stats['errors']}")

    print("\nTest directory:")
    print(f"   Total images found: {test_stats['total']}")
    print(f"   Images matching patterns: {test_stats['matched']}")
    print(f"   Images copied: {test_stats['copied']}")
    print(f"   Errors: {test_stats['errors']}")

    total_copied = train_stats['copied'] + test_stats['copied']
    total_errors = train_stats['errors'] + test_stats['errors']

    print("\nOverall:")
    print(f"   Total images copied: {total_copied}")
    print(f"   Total errors: {total_errors}")

    if total_errors == 0:
        print("\nAll images copied successfully!")
    else:
        print(
            "\nSome errors occurred during copying. Check the output above."
        )


def _process_directory(
    source_dir: Path,
    target_dir: Path,
    name_patterns: List[str],
    image_extensions: Set[str],
    dir_type: str
) -> dict:
    """
    Process a single directory (train or test) and copy matching images.

    Parameters
    ----------
    source_dir : Path
        Source directory to process
    target_dir : Path
        Target directory to copy matching images to
    name_patterns : List[str]
        List of patterns to match in image names
    image_extensions : Set[str]
        Set of supported image file extensions
    dir_type : str
        Type of directory being processed (for display purposes)

    Returns
    -------
    dict
        Dictionary with statistics about the processing containing keys:
        'total', 'matched', 'copied', 'errors'
    """
    stats = {
        'total': 0,
        'matched': 0,
        'copied': 0,
        'errors': 0
    }

    print(f"\nProcessing {dir_type} directory: {source_dir}")

    # Get all image files
    image_files = []
    for file_path in source_dir.iterdir():
        if (file_path.is_file() and
                file_path.suffix.lower() in image_extensions):
            image_files.append(file_path)

    stats['total'] = len(image_files)
    print(f"   Found {stats['total']} image files")

    if stats['total'] == 0:
        print(f"   Warning: No image files found in {source_dir}")
        return stats

    # Filter images based on name patterns
    matching_files = []
    for file_path in image_files:
        file_name = file_path.name.lower()
        if any(pattern.lower() in file_name
               for pattern in name_patterns):
            matching_files.append(file_path)

    stats['matched'] = len(matching_files)
    print(f"   {stats['matched']} images match the patterns")

    if stats['matched'] == 0:
        print("   Warning: No images match the provided patterns")
        return stats

    # Copy matching files
    print(f"   Copying {stats['matched']} files...")
    for file_path in matching_files:
        try:
            target_file = target_dir / file_path.name
            shutil.copy2(file_path, target_file)
            stats['copied'] += 1
            print(f"   OK: {file_path.name}")
        except Exception as e:
            stats['errors'] += 1
            print(f"   Error copying {file_path.name}: {e}")

    return stats


def get_user_input() -> tuple:
    """
    Get user input for the script parameters.

    Returns
    -------
    tuple
        Tuple containing (train_dir, test_dir, name_patterns, output_dir)
    """
    print("=" * 60)
    print("IMAGE FILTERING AND COPYING TOOL")
    print("=" * 60)
    print("This script will filter images from train/test directories")
    print("based on partial name matches and copy them to an output "
          "directory.")
    print()

    # Get input directories
    train_dir = input("Enter path to train directory: ").strip().strip('"\'')
    test_dir = input("Enter path to test directory: ").strip().strip('"\'')
    # Get name patterns
    print("\nEnter name patterns to match (partial matches, "
          "case-insensitive):")
    print("You can enter multiple patterns separated by commas.")
    print("Example: 'cell', 'nucleus', 'mitochondria'")
    patterns_input = input("Patterns: ").strip()

    if not patterns_input:
        raise ValueError("No patterns provided")

    name_patterns = [
        pattern.strip() for pattern in patterns_input.split(',')
        if pattern.strip()
    ]

    # Get output directory
    output_dir = input("\nEnter output directory path: ").strip().strip('"\'')

    return train_dir, test_dir, name_patterns, output_dir


def main():
    """Main function to run the script."""
    try:
        train_dir, test_dir, name_patterns, output_dir = get_user_input()

        print("\nConfiguration:")
        print(f"   Train directory: {train_dir}")
        print(f"   Test directory: {test_dir}")
        print(f"   Name patterns: {name_patterns}")
        print(f"   Output directory: {output_dir}")

        confirm = input(
            "\nProceed with filtering and copying? (y/n): "
        ).lower().strip()
        if confirm not in ['y', 'yes']:
            print("Operation cancelled.")
            return

        filter_and_copy_images(train_dir, test_dir, name_patterns, output_dir)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
