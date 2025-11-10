#!/usr/bin/env python3
"""
Organize files into directories based on their prefix (AS10, AS11, etc.)
"""

import argparse
import shutil
from pathlib import Path
import re


def extract_prefix(filename):
    """Extract the AS prefix from filename (e.g., AS10, AS11, AS2, etc.)"""
    match = re.match(r'^([a-f0-9]+-)?([A-Z]+\d+[A-Z]?)_', filename)
    if match:
        return match.group(2)
    return None


def organize_files(input_dir, output_dir, dry_run=False):
    """
    Organize files from input_dir into subdirectories in output_dir.
    
    Args:
        input_dir: Source directory containing files
        output_dir: Destination directory for organized files
        dry_run: If True, only show what would be done without copying
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    files = list(input_path.glob('*'))
    if not files:
        print(f"No files found in '{input_dir}'")
        return
    
    print(f"Processing {len(files)} files from '{input_dir}'")
    
    stats = {}
    skipped = 0
    
    for file_path in files:
        if not file_path.is_file():
            continue
            
        prefix = extract_prefix(file_path.name)
        
        if not prefix:
            print(f"Skipping: {file_path.name} (no valid prefix)")
            skipped += 1
            continue
        
        target_dir = output_path / prefix
        target_file = target_dir / file_path.name
        
        stats[prefix] = stats.get(prefix, 0) + 1
        
        if dry_run:
            print(f"Would copy: {file_path.name} -> {prefix}/")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_file)
    
    # Summary
    print(f"\n{'DRY RUN - ' if dry_run else ''}Summary:")
    for prefix in sorted(stats.keys()):
        print(f"  {prefix}: {stats[prefix]} files")
    
    if skipped:
        print(f"  Skipped: {skipped} files")
    
    print(f"\nTotal: {sum(stats.values())} files {'would be' if dry_run else ''} organized")


def main():
    parser = argparse.ArgumentParser(
        description='Organize files into directories based on their prefix (AS10, AS11, etc.)'
    )
    parser.add_argument(
        '--input_dir',
        help='Input directory containing files to organize'
    )
    parser.add_argument(
        '--output_dir',
        help='Output directory where organized subdirectories will be created'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    
    args = parser.parse_args()
    
    organize_files(args.input_dir, args.output_dir, args.dry_run)


if __name__ == '__main__':
    main()