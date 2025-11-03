#!/usr/bin/env python3
"""
Script to filter microscope image files based on magnification values in filenames.

Filename format examples
------------------------
- 0ab7de9d-AS2_40_10_jpg.rf.209a8405481b2434b8436c3f3acd60fd_test_0204_image.jpg
- AS3_40_17_jpg.rf.cd63d6404ff46a71b4f34303e14ea5e8_test_0058_masks.tif

Magnification is extracted from the pattern _XX_ where XX is the magnification value.

Examples
--------
Filter files with magnifications 40 and 50:
    $ python filter_magnification.py --source ./images --magnifications 40 50

Dry run to preview changes:
    $ python filter_magnification.py --source ./images --magnifications 40 50 --dry-run

Copy filtered files to output directory:
    $ python filter_magnification.py --source ./images --output ./filtered --magnifications 40 50 --copy
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Set, Optional


def extract_magnification(filename: str) -> Optional[int]:
    """
    Extract magnification value from filename.
    
    Looks for pattern _NUMBER_ in the filename and returns the first match.
    
    Parameters
    ----------
    filename : str
        The filename to parse.
        
    Returns
    -------
    int or None
        Magnification value as integer, or None if not found.
        
    Examples
    --------
    >>> extract_magnification("AS2_40_10_jpg.rf.test_0204_image.jpg")
    40
    >>> extract_magnification("AS3_100_17_jpg.rf.test_0058_masks.tif")
    100
    >>> extract_magnification("no_magnification_here.jpg")
    None
    """
    # Pattern to match _NUMBER_ in filename
    # This looks for underscore, digits, underscore
    pattern = r'_(\d+)_'
    matches = re.findall(pattern, filename)
    
    # Return the first match (should be the magnification based on your examples)
    # In your examples, the magnification appears as the first _XX_ pattern
    if matches:
        return int(matches[0])
    return None


def filter_files_by_magnification(
    directory: str,
    allowed_magnifications: Set[int],
    output_dir: Optional[str] = None,
    copy_files: bool = False,
    dry_run: bool = True
) -> List[str]:
    """
    Filter files based on magnification values.
    
    Parameters
    ----------
    directory : str
        Directory containing the files to filter.
    allowed_magnifications : set of int
        Set of magnification values to keep (e.g., {40, 50}).
    output_dir : str, optional
        Output directory to copy/move filtered files. If None, files remain
        in place (default is None).
    copy_files : bool, optional
        If True, copy files; if False, move files. Only used when output_dir
        is specified (default is False).
    dry_run : bool, optional
        If True, only print what would be done without actually performing
        any file operations (default is True).
        
    Returns
    -------
    list of str
        List of filtered filenames that matched the magnification criteria.
        
    Examples
    --------
    >>> mags = {40, 50}
    >>> filtered = filter_files_by_magnification("./images", mags, dry_run=True)
    >>> len(filtered)
    10
    """
    directory_path = Path(directory)
    filtered_files = []
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return filtered_files
    
    # Create output directory if specified and not in dry run mode
    if output_dir and not dry_run:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all files in directory
    for file_path in directory_path.iterdir():
        if file_path.is_file():
            filename = file_path.name
            magnification = extract_magnification(filename)
            
            if magnification in allowed_magnifications:
                filtered_files.append(filename)
                
                if dry_run:
                    print(f"[DRY RUN] Would process: {filename} (magnification: {magnification})")
                else:
                    print(f"Processing: {filename} (magnification: {magnification})")
                    
                    if output_dir:
                        output_path = Path(output_dir) / filename
                        if copy_files:
                            shutil.copy2(file_path, output_path)
                            print(f"  -> Copied to {output_path}")
                        else:
                            shutil.move(str(file_path), str(output_path))
                            print(f"  -> Moved to {output_path}")
            else:
                if magnification:
                    print(f"Skipping: {filename} (magnification: {magnification} - not in allowed list)")
                else:
                    print(f"Skipping: {filename} (magnification not found)")
    
    return filtered_files


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Filter microscope image files by magnification values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview) with magnifications 40 and 50
  python filter_magnification.py --source ./images --magnifications 40 50 --dry-run
  
  # Copy files with magnification 40 to output directory
  python filter_magnification.py --source ./images --output ./filtered --magnifications 40 --copy
  
  # Move files with magnifications 40, 50, and 60 (no dry run)
  python filter_magnification.py --source ./images --output ./filtered --magnifications 40 50 60
        """
    )
    
    parser.add_argument(
        '-s', '--source',
        type=str,
        required=True,
        help='Source directory containing the image files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for filtered files (if not specified, files stay in place)'
    )
    
    parser.add_argument(
        '-m', '--magnifications',
        type=int,
        nargs='+',
        required=True,
        help='Magnification values to keep (e.g., 40 50 60)'
    )
    
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them (only applies when --output is specified)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without actually performing file operations'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """
    Main function with command-line argument parsing.
    
    Processes command-line arguments and filters microscope image files
    based on specified magnification values.
    """
    args = parse_arguments()
    
    # Convert magnifications list to set
    allowed_magnifications = set(args.magnifications)
    
    print("=" * 70)
    print("Microscope Image Magnification Filter")
    print("=" * 70)
    print(f"Source directory: {args.source}")
    print(
        f"Output directory: {args.output if args.output else 'N/A (filtering in place)'}")
    print(f"Allowed magnifications: {sorted(allowed_magnifications)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    if args.output:
        print(f"Action: {'COPY' if args.copy else 'MOVE'}")
    print("=" * 70)
    print()
    
    # Run the filter
    filtered = filter_files_by_magnification(
        directory=args.source,
        allowed_magnifications=allowed_magnifications,
        output_dir=args.output,
        copy_files=args.copy,
        dry_run=args.dry_run
    )
    
    print()
    print("=" * 70)
    print(f"Summary: {len(filtered)} files matched the filter")
    print("=" * 70)
    
    if args.dry_run:
        print(
            "\n⚠️  This was a DRY RUN - no files were actually moved or copied.")
        print("Run without --dry-run flag to perform the actual operation.")
    else:
        print("\n✓ Operation completed successfully!")


if __name__ == "__main__":
    main()