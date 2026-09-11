"""Repoint stale symlinks in sourcedata/ and derivatives/ from the old
parietal store path to the neurospin mount.
"""

import argparse
from pathlib import Path

OLD_BASE = "/data/parietal/store2/data/ibc/"
NEW_BASE = "/neurospin/ibc/"


def find_stale_links(roots: list[Path]):
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_symlink():
                continue
            target = str(path.readlink())
            if target.startswith(OLD_BASE):
                yield path, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Directories to scan (e.g. sourcedata derivatives)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would change")
    args = parser.parse_args()

    n_fixed = 0
    for path, target in find_stale_links(args.roots):
        new_target = NEW_BASE + target[len(OLD_BASE):]
        print(f"{path} -> {new_target}")
        if not args.dry_run:
            path.unlink()
            path.symlink_to(new_target)
        n_fixed += 1

    print(f"{'Would fix' if args.dry_run else 'Fixed'} {n_fixed} symlink(s).")


if __name__ == "__main__":
    main()
