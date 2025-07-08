#!/usr/bin/env python3
"""
Normalize TOC structure and fix missing hierarchical entries
"""
import re
import sys
from pathlib import Path


def extract_entries(content):
    """Extract all TOC entries with their hierarchical IDs"""
    entries = []
    
    # Pattern: number sequence + space + title
    pattern = r'^(\d+(?:\.\d+)*\.?)\s+(.+)$'
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        match = re.match(pattern, line)
        if match:
            hierarchical_id = match.group(1).rstrip('.')  # Remove trailing dot
            title = match.group(2).strip()
            entries.append((hierarchical_id, title))
    
    return entries


def generate_missing_parents(entries):
    """Generate missing parent entries"""
    existing_ids = {entry[0] for entry in entries}
    missing_entries = []
    
    for hierarchical_id, title in entries:
        parts = hierarchical_id.split('.')
        
        # Generate all parent paths
        for i in range(1, len(parts)):
            parent_id = '.'.join(parts[:i])
            
            if parent_id not in existing_ids:
                # Generate generic title for missing parent
                level_names = ['Rozdział', 'Sekcja', 'Podsekcja', 'Punkt']
                level = min(i-1, len(level_names)-1)
                parent_title = f"{level_names[level]} {parent_id}"
                
                missing_entries.append((parent_id, parent_title))
                existing_ids.add(parent_id)
    
    return missing_entries


def sort_hierarchical(entries):
    """Sort entries by hierarchical order"""
    def sort_key(entry):
        hierarchical_id = entry[0]
        parts = hierarchical_id.split('.')
        return [int(part) for part in parts]
    
    return sorted(entries, key=sort_key)


def format_toc(entries):
    """Format entries back to TOC format"""
    lines = []
    
    for hierarchical_id, title in entries:
        lines.append(f"{hierarchical_id} {title}")
    
    return '\n'.join(lines)


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_toc.py <toc_file>")
        sys.exit(1)
    
    toc_file = Path(sys.argv[1])
    
    if not toc_file.exists():
        print(f"Error: File {toc_file} not found")
        sys.exit(1)
    
    # Read with proper encoding detection
    encodings = ['utf-8', 'cp1250', 'iso-8859-2']
    content = None
    
    for encoding in encodings:
        try:
            with open(toc_file, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"📖 Successfully read with {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        print("❌ Could not decode file with any encoding")
        sys.exit(1)
    
    print(f"📝 Processing {toc_file}")
    
    # Extract entries
    entries = extract_entries(content)
    print(f"📊 Found {len(entries)} entries")
    
    # Generate missing parents
    missing_entries = generate_missing_parents(entries)
    if missing_entries:
        print(f"➕ Adding {len(missing_entries)} missing parent entries")
        entries.extend(missing_entries)
    
    # Sort all entries
    entries = sort_hierarchical(entries)
    
    # Format and save with UTF-8
    normalized_content = format_toc(entries)
    
    with open(toc_file, 'w', encoding='utf-8') as f:
        f.write(normalized_content)
    
    print(f"✅ Normalized TOC saved with {len(entries)} total entries")


if __name__ == "__main__":
    main()