import os
import csv


class CSVLogger:
    """
    Lightweight per-epoch CSV logger.

    Creates the file and writes a header row on first call to log().
    Appends one row per subsequent call — safe to use across interrupted
    training runs (will append to existing file).

    Args:
        filepath (str): path to the .csv log file
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self._header_written = os.path.isfile(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def log(self, row: dict):
        """
        Write one row to the CSV.

        Args:
            row (dict): keys become column headers on first write
        """
        write_header = not self._header_written
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def read(self):
        """
        Read the log back as a list of dicts.

        Returns:
            list[dict]: one entry per logged row
        """
        if not os.path.isfile(self.filepath):
            return []
        with open(self.filepath, "r") as f:
            return list(csv.DictReader(f))