"""Results go to reports/ as CSV, so a run leaves something behind besides scrollback."""
import csv
from pathlib import Path

REPORTS = Path("reports")


def write_report(name, columns, rows):
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{name}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"\nwrote {path}")
