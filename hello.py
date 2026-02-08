#!/usr/bin/env python3
"""Simple greeting script that prints a message with the current date.

The date is formatted as YYYY-MM-DD.
"""

from datetime import datetime

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Hello! Today's date is {today}.")


if __name__ == "__main__":
    main()

