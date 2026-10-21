"""Aligned columns, so the other files can be about their subject instead of padding."""


def table(headers, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    for row in [headers, *rows]:
        print("  ".join(str(cell).ljust(width) if index == 0 else str(cell).rjust(width)
                        for index, (cell, width) in enumerate(zip(row, widths))))
