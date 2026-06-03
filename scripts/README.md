# Scripts

Utility scripts for the EarthquakesParser project.

## Available Scripts

### [compute_credibility.py](compute_credibility.py)

Computes consensus-based credibility scores for all records in Tier 1 Milvus collection.

**Usage:**

```bash
# Use default parameters (recommended)
python scripts/compute_credibility.py

# High similarity threshold (stricter grouping)
python scripts/compute_credibility.py --threshold 0.90

# Require larger groups
python scripts/compute_credibility.py --min-group-size 5

# Adjust consensus/detail balance
python scripts/compute_credibility.py --centrality-weight 0.7 --detail-weight 0.3

# Different collection
python scripts/compute_credibility.py --collection veritatis_tier2_arena

# Verbose logging
python scripts/compute_credibility.py --verbose
```

**What it does:**

- Finds groups of similar articles using vector similarity
- Ranks articles within groups by centrality (consensus)
- Assigns credibility scores (0-1) based on consensus strength
- Records without similar neighbors get neutral score (0.5)
- Updates all credibility_score fields in Milvus

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--collection` | veritatis_tier1_lake | Milvus collection name |
| `--threshold` | 0.85 | Minimum similarity (0-1) to group records |
| `--min-group-size` | 2 | Minimum records per group |
| `--centrality-weight` | 0.6 | Weight for consensus |
| `--detail-weight` | 0.4 | Weight for text quality |
| `--neutral-score` | 0.5 | Score for singleton records |
| `--verbose` | false | Enable debug logging |

**Example output:**

```text
======================================================================
Credibility Score Computation
======================================================================

Configuration:
  Collection:          veritatis_tier1_lake
  Similarity threshold: 0.85
  Min group size:      2
  Centrality weight:   0.6
  Detail weight:       0.4
  Neutral score:       0.5

======================================================================
Results
======================================================================

✓ Computation complete!

Statistics:
  Total records:         156
  Groups found:          23
  Records in groups:     89
  Records without groups: 67
  Updated count:         156

  Average group size:    3.9 records
  Group coverage:        57.1%

✓ Successfully updated 156 credibility scores

======================================================================
```

### [bump_version.py](bump_version.py)

Automatically bumps project version and updates CHANGELOG based on conventional commits.

**Usage:**

```bash
# Auto-detect version bump from commits
python scripts/bump_version.py

# Specify bump type manually
python scripts/bump_version.py --type minor

# Dry run (preview changes)
python scripts/bump_version.py --dry-run

# Skip tag creation
python scripts/bump_version.py --no-tag
```

**What it does:**

- Analyzes commits since last tag
- Determines version bump type (major/minor/patch)
- Updates version in `pyproject.toml` and `__init__.py`
- Updates `CHANGELOG.md` with categorized commits
- Creates git tag for new version

**Version bump rules:**

- `feat:` commits → **minor** version bump
- `fix:` commits → **patch** version bump
- `BREAKING CHANGE:` or `!` → **major** version bump

### [verify_setup.py](verify_setup.py)

Verifies that the project is properly set up with all required files and structure.

**Usage:**

```bash
python scripts/verify_setup.py
```

**What it checks:**

- Package structure (earthquakes_parser/, tests/, etc.)
- Configuration files (pyproject.toml, .flake8, etc.)
- Documentation files
- CI/CD workflows
- Module imports (if package is installed)

**Example output:**

```text
🔍 EarthquakesParser Setup Verification

📁 Checking package structure...
✅ Main package: earthquakes_parser
✅ Search module: earthquakes_parser/search
...

🎉 All checks passed! (24/24)
```

## Running Scripts

### From project root

```bash
python scripts/verify_setup.py
```

### Make executable (Unix/macOS)

```bash
chmod +x scripts/verify_setup.py
./scripts/verify_setup.py
```

## Adding New Scripts

When adding utility scripts:

1. Place them in this `scripts/` directory
2. Add a shebang line: `#!/usr/bin/env python3`
3. Document them in this README
4. Make them executable if needed
