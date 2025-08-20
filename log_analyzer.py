#!/usr/bin/env python3
"""
Log File Analyzer (OOP) — parses .log files, summarizes, filters, and visualizes.

Features
- Supports common timestamp formats and levels: INFO, WARNING/WARN, ERROR (DEBUG tolerated but ignored by default).
- Streams large files line-by-line, handles malformed lines gracefully.
- Filtering by keyword and date range.
- Summary stats (total, per level, first/last timestamp, common errors).
- Exports JSON results.
- Matplotlib visualizations: bar chart (counts by level) and timeline (frequency over time).
- CLI for automation; minimal Tkinter GUI for interactive use.

Usage (CLI)
    python log_analyzer.py analyze path/to/app.log \
        --levels INFO WARNING ERROR \
        --from 2025-08-20T00:00:00 --to 2025-08-20T23:59:59 \
        --keyword "disk" \
        --json-out results.json \
        --plot both --show

    python log_analyzer.py gui

Notes
- Date/time arguments accept ISO-8601 like "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS".
- Timeline uses automatic binning (minute resolution by default) and can be adjusted in code.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Iterable, Iterator, List, Optional, Dict, Tuple

# Matplotlib is imported lazily inside the visualizer to keep CLI light if no plots requested.


# ------------------------------
# Data model
# ------------------------------
@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    message: str
    raw_line: Optional[str] = None


# ------------------------------
# Parsing
# ------------------------------
class LogParser:
    """Parses text .log files into LogEntry objects using regex patterns.

    The parser tries multiple patterns to accommodate common log formats.
    Add patterns as needed.
    """

    # Common regex patterns. Named groups: ts, level, msg
    _PATTERNS: List[re.Pattern] = [
        # 2025-08-20 14:22:15 INFO Message
        re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)\s+(?P<level>INFO|WARNING|WARN|ERROR|DEBUG)\s+(?P<msg>.*)$"),
        # 2025-08-20T14:22:15Z [INFO] Message
        re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?)\s*(?:\[(?P<level>INFO|WARNING|WARN|ERROR|DEBUG)\])\s+(?P<msg>.*)$"),
        # [2025-08-20 14:22:15] ERROR Message
        re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)\]\s+(?P<level>INFO|WARNING|WARN|ERROR|DEBUG)\s+(?P<msg>.*)$"),
        # INFO 2025-08-20 14:22:15 Message
        re.compile(r"^(?P<level>INFO|WARNING|WARN|ERROR|DEBUG)\s+(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)\s+(?P<msg>.*)$"),
    ]

    # Datetime parse formats to try (aligned with patterns above)
    _DT_FORMATS: Tuple[str, ...] = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    )

    def __init__(self, filepath: str, include_levels: Optional[List[str]] = None):
        self.filepath = filepath
        self.include_levels = [lvl.upper() for lvl in include_levels] if include_levels else ["INFO", "WARNING", "ERROR"]
        self.malformed_count = 0
        self.total_lines = 0

    def _parse_timestamp(self, s: str) -> Optional[datetime]:
        s = s.replace(",", ".")  # allow comma as millisecond separator
        # Strip trailing Z or timezone offset (basic support)
        s_noz = re.sub(r"(Z|[+-]\d{2}:?\d{2})$", "", s)
        for fmt in self._DT_FORMATS:
            try:
                return datetime.strptime(s_noz, fmt)
            except ValueError:
                continue
        return None

    def parse(self) -> Iterator[LogEntry]:
        """Yield LogEntry objects by streaming the file line-by-line."""
        with open(self.filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                self.total_lines += 1
                line = line.rstrip("\n")
                entry = self._parse_line(line)
                if entry is None:
                    self.malformed_count += 1
                    continue
                if entry.level not in self.include_levels:
                    # Skip levels outside requested set
                    continue
                yield entry

    def _parse_line(self, line: str) -> Optional[LogEntry]:
        for pat in self._PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            ts_raw = m.group("ts")
            level = m.group("level").replace("WARN", "WARNING").upper()
            msg = m.group("msg").strip()
            ts = self._parse_timestamp(ts_raw)
            if ts is None:
                return None
            return LogEntry(timestamp=ts, level=level, message=msg, raw_line=line)
        return None


# ------------------------------
# Analysis
# ------------------------------
class LogAnalyzer:
    def __init__(self, entries: Iterable[LogEntry]):
        self._entries: List[LogEntry] = list(entries)
        self._entries.sort(key=lambda e: e.timestamp)
        self._index_built = False
        self._by_level: Dict[str, List[LogEntry]] = defaultdict(list)

    # --- Indexing helpers ---
    def _ensure_index(self):
        if self._index_built:
            return
        for e in self._entries:
            self._by_level[e.level].append(e)
        self._index_built = True

    # --- Basic stats ---
    @property
    def total(self) -> int:
        return len(self._entries)

    def counts_by_level(self) -> Dict[str, int]:
        self._ensure_index()
        return {level: len(lst) for level, lst in self._by_level.items()}

    def first_last(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        if not self._entries:
            return None, None
        return self._entries[0].timestamp, self._entries[-1].timestamp

    def most_common_errors(self, n: int = 5) -> List[Tuple[str, int]]:
        self._ensure_index()
        msgs = [e.message for e in self._by_level.get("ERROR", [])]
        return Counter(msgs).most_common(n)

    # --- Filtering ---
    def filter(
        self,
        keyword: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        levels: Optional[List[str]] = None,
    ) -> List[LogEntry]:
        levels_set = set([lvl.upper() for lvl in levels]) if levels else None
        kw = keyword.lower() if keyword else None
        out: List[LogEntry] = []
        for e in self._entries:
            if levels_set and e.level not in levels_set:
                continue
            if start and e.timestamp < start:
                continue
            if end and e.timestamp > end:
                continue
            if kw and kw not in e.message.lower():
                continue
            out.append(e)
        return out

    # --- Summary dict for JSON export ---
    def summary(self, top_n_errors: int = 5) -> Dict:
        counts = self.counts_by_level()
        first, last = self.first_last()
        return {
            "total_entries": self.total,
            "counts_by_level": counts,
            "first_timestamp": first.isoformat() if first else None,
            "last_timestamp": last.isoformat() if last else None,
            "common_errors": [
                {"message": msg, "count": cnt} for msg, cnt in self.most_common_errors(top_n_errors)
            ],
        }


# ------------------------------
# Visualization
# ------------------------------
class LogVisualizer:
    def __init__(self):
        # Lazy import to avoid requiring matplotlib unless needed
        global plt
        import matplotlib.pyplot as plt  # type: ignore

    def bar_counts(self, counts: Dict[str, int], title: str = "Log counts by level", save_path: Optional[str] = None, show: bool = True):
        labels = list(counts.keys())
        values = [counts[k] for k in labels]
        plt.figure()
        plt.bar(labels, values)
        plt.title(title)
        plt.xlabel("Level")
        plt.ylabel("Count")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def timeline(self, entries: List[LogEntry], bin_size: str = "minute", title: str = "Log timeline", save_path: Optional[str] = None, show: bool = True):
        if not entries:
            return
        # Bin by minute/hour depending on bin_size
        def truncate(dt: datetime) -> datetime:
            if bin_size == "hour":
                return dt.replace(minute=0, second=0, microsecond=0)
            else:  # minute
                return dt.replace(second=0, microsecond=0)

        counts: Dict[datetime, int] = defaultdict(int)
        for e in entries:
            counts[truncate(e.timestamp)] += 1
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]

        plt.figure()
        plt.plot(xs, ys)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Entries per %s" % ("hour" if bin_size == "hour" else "minute"))
        plt.gcf().autofmt_xdate()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


# ------------------------------
# CLI + Orchestration
# ------------------------------
class LogApp:
    def __init__(self):
        pass

    @staticmethod
    def _parse_date_arg(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        s = s.strip()
        # Accept YYYY-MM-DD or full datetime
        fmts = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]
        for fmt in fmts:
            try:
                if fmt == "%Y-%m-%d":
                    d = datetime.strptime(s, fmt)
                    return d  # interpreted as midnight
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise argparse.ArgumentTypeError(f"Invalid date/datetime format: {s}")

    def run_cli(self, argv: Optional[List[str]] = None) -> int:
        parser = argparse.ArgumentParser(description="Log File Analyzer (OOP)")
        sub = parser.add_subparsers(dest="command", required=True)

        p_an = sub.add_parser("analyze", help="Analyze a .log file")
        p_an.add_argument("logfile", help="Path to .log file")
        p_an.add_argument("--levels", nargs="*", default=["INFO", "WARNING", "ERROR"], help="Levels to include (default: INFO WARNING ERROR)")
        p_an.add_argument("--from", dest="from_dt", type=self._parse_date_arg, help="Start date/time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        p_an.add_argument("--to", dest="to_dt", type=self._parse_date_arg, help="End date/time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        p_an.add_argument("--keyword", help="Keyword to search (case-insensitive)")
        p_an.add_argument("--json-out", dest="json_out", help="Write JSON summary to path")
        p_an.add_argument("--plot", choices=["none", "bar", "timeline", "both"], default="none", help="Which plots to generate")
        p_an.add_argument("--show", action="store_true", help="Show plots interactively")
        p_an.add_argument("--save-prefix", help="Save plot images with this path prefix (e.g., out/plot)")
        p_an.add_argument("--timeline-bin", choices=["minute", "hour"], default="minute", help="Aggregation for timeline plot")

        p_gui = sub.add_parser("gui", help="Open a minimal GUI for analysis")

        args = parser.parse_args(argv)

        if args.command == "gui":
            return self._run_gui()

        # Analyze command
        logfile = args.logfile
        if not os.path.isfile(logfile):
            print(f"File not found: {logfile}", file=sys.stderr)
            return 2

        parser_obj = LogParser(logfile, include_levels=args.levels)
        entries = list(parser_obj.parse())
        analyzer = LogAnalyzer(entries)

        # Apply additional filtering (keyword/date)
        filtered = analyzer.filter(
            keyword=args.keyword,
            start=args.from_dt,
            end=args.to_dt,
            levels=args.levels,
        )

        # Print quick summary to stdout
        counts = Counter([e.level for e in filtered])
        first, last = (filtered[0].timestamp, filtered[-1].timestamp) if filtered else (None, None)
        print("--- Analysis Summary ---")
        print(f"Log file: {logfile}")
        print(f"Included levels: {', '.join(args.levels)}")
        if args.keyword:
            print(f"Keyword filter: {args.keyword}")
        if args.from_dt or args.to_dt:
            print(f"Date range: {args.from_dt or '-'} to {args.to_dt or '-'}")
        print(f"Total parsed (matching filters): {len(filtered)}")
        print("Counts by level:")
        for lvl in ["ERROR", "WARNING", "INFO"]:
            if counts.get(lvl):
                print(f"  {lvl}: {counts[lvl]}")
        if first and last:
            print(f"First timestamp: {first.isoformat()}")
            print(f"Last  timestamp: {last.isoformat()}")

        # JSON export (use overall analyzer summary + note filters)
        if args.json_out:
            summary = analyzer.summary()
            summary.update({
                "filters": {
                    "levels": args.levels,
                    "keyword": args.keyword,
                    "from": args.from_dt.isoformat() if args.from_dt else None,
                    "to": args.to_dt.isoformat() if args.to_dt else None,
                },
                "file": os.path.abspath(logfile),
                "malformed_lines": parser_obj.malformed_count,
                "total_lines": parser_obj.total_lines,
                "filtered_count": len(filtered),
            })
            with open(args.json_out, "w", encoding="utf-8") as jf:
                json.dump(summary, jf, indent=2, ensure_ascii=False)
            print(f"JSON written to: {args.json_out}")

        # Plots
        if args.plot != "none":
            viz = LogVisualizer()
            save_prefix = args.save_prefix
            if args.plot in ("bar", "both"):
                save_path = f"{save_prefix}_bar.png" if save_prefix else None
                viz.bar_counts(dict(counts), save_path=save_path, show=args.show)
                if save_path:
                    print(f"Bar chart saved to: {save_path}")
            if args.plot in ("timeline", "both"):
                save_path = f"{save_prefix}_timeline.png" if save_prefix else None
                viz.timeline(filtered, bin_size=args.timeline_bin, save_path=save_path, show=args.show)
                if save_path:
                    print(f"Timeline chart saved to: {save_path}")

        # Exit code indicates success and whether malformed lines were encountered
        if parser_obj.malformed_count > 0:
            print(f"Note: {parser_obj.malformed_count} malformed lines were skipped.")
        return 0

    # ------------------------------
    # Minimal Tkinter GUI
    # ------------------------------
    def _run_gui(self) -> int:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except Exception as e:
            print("Tkinter is not available in this environment.", file=sys.stderr)
            return 3

        class App(tk.Tk):
            def __init__(self):
                super().__init__()
                self.title("Log File Analyzer")
                self.geometry("700x500")

                self.log_path_var = tk.StringVar()
                self.keyword_var = tk.StringVar()
                self.from_var = tk.StringVar()
                self.to_var = tk.StringVar()
                self.level_vars = {lvl: tk.BooleanVar(value=True) for lvl in ("INFO", "WARNING", "ERROR")}

                frm = tk.Frame(self)
                frm.pack(fill=tk.X, padx=10, pady=10)

                # File chooser
                tk.Label(frm, text="Log file:").grid(row=0, column=0, sticky="w")
                tk.Entry(frm, textvariable=self.log_path_var, width=60).grid(row=0, column=1, padx=5)
                tk.Button(frm, text="Browse", command=self.browse).grid(row=0, column=2)

                # Filters
                tk.Label(frm, text="Keyword:").grid(row=1, column=0, sticky="w")
                tk.Entry(frm, textvariable=self.keyword_var).grid(row=1, column=1, sticky="we", padx=5)

                tk.Label(frm, text="From (YYYY-MM-DD or T time):").grid(row=2, column=0, sticky="w")
                tk.Entry(frm, textvariable=self.from_var).grid(row=2, column=1, sticky="we", padx=5)

                tk.Label(frm, text="To (YYYY-MM-DD or T time):").grid(row=3, column=0, sticky="w")
                tk.Entry(frm, textvariable=self.to_var).grid(row=3, column=1, sticky="we", padx=5)

                # Levels
                lvl_frame = tk.Frame(frm)
                lvl_frame.grid(row=4, column=1, sticky="w", pady=5)
                for i, lvl in enumerate(["INFO", "WARNING", "ERROR"]):
                    tk.Checkbutton(lvl_frame, text=lvl, variable=self.level_vars[lvl]).grid(row=0, column=i)

                # Buttons
                btn_frame = tk.Frame(frm)
                btn_frame.grid(row=5, column=1, sticky="w", pady=10)
                tk.Button(btn_frame, text="Analyze", command=self.analyze).grid(row=0, column=0, padx=5)
                tk.Button(btn_frame, text="Bar Chart", command=lambda: self.plot("bar")).grid(row=0, column=1, padx=5)
                tk.Button(btn_frame, text="Timeline", command=lambda: self.plot("timeline")).grid(row=0, column=2, padx=5)

                # Output
                self.out = tk.Text(self, height=18)
                self.out.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                self.entries: List[LogEntry] = []

            def browse(self):
                path = filedialog.askopenfilename(title="Select log file", filetypes=[("Log files", "*.log"), ("All files", "*.*")])
                if path:
                    self.log_path_var.set(path)

            def parse_dt(self, s: str) -> Optional[datetime]:
                s = s.strip()
                if not s:
                    return None
                for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        d = datetime.strptime(s, fmt)
                        return d
                    except ValueError:
                        continue
                messagebox.showerror("Invalid date", f"Invalid date/datetime: {s}")
                return None

            def analyze(self):
                path = self.log_path_var.get()
                if not path or not os.path.isfile(path):
                    messagebox.showerror("Error", "Please select a valid log file.")
                    return
                levels = [lvl for lvl, var in self.level_vars.items() if var.get()]
                parser_obj = LogParser(path, include_levels=levels)
                self.entries = list(parser_obj.parse())
                analyzer = LogAnalyzer(self.entries)
                start = self.parse_dt(self.from_var.get())
                end = self.parse_dt(self.to_var.get())
                keyword = self.keyword_var.get().strip() or None
                filtered = analyzer.filter(keyword=keyword, start=start, end=end, levels=levels)

                counts = Counter([e.level for e in filtered])
                first, last = (filtered[0].timestamp, filtered[-1].timestamp) if filtered else (None, None)
                self.out.delete("1.0", tk.END)
                self.out.insert(tk.END, f"File: {path}\n")
                self.out.insert(tk.END, f"Levels: {', '.join(levels)}\n")
                if keyword:
                    self.out.insert(tk.END, f"Keyword: {keyword}\n")
                if start or end:
                    self.out.insert(tk.END, f"Date range: {start or '-'} to {end or '-'}\n")
                self.out.insert(tk.END, f"Total entries: {len(filtered)}\n")
                self.out.insert(tk.END, "Counts by level:\n")
                for lvl in ["ERROR", "WARNING", "INFO"]:
                    if counts.get(lvl):
                        self.out.insert(tk.END, f"  {lvl}: {counts[lvl]}\n")
                if first and last:
                    self.out.insert(tk.END, f"First: {first.isoformat()}\nLast:  {last.isoformat()}\n")
                if parser_obj.malformed_count:
                    self.out.insert(tk.END, f"Note: {parser_obj.malformed_count} malformed lines were skipped.\n")

            def plot(self, which: str):
                if not self.entries:
                    messagebox.showinfo("Info", "Run Analyze first.")
                    return
                analyzer = LogAnalyzer(self.entries)
                keyword = self.keyword_var.get().strip() or None
                start = self.parse_dt(self.from_var.get())
                end = self.parse_dt(self.to_var.get())
                levels = [lvl for lvl, var in self.level_vars.items() if var.get()]
                filtered = analyzer.filter(keyword=keyword, start=start, end=end, levels=levels)
                viz = LogVisualizer()
                if which == "bar":
                    counts = Counter([e.level for e in filtered])
                    viz.bar_counts(dict(counts), show=True)
                else:
                    viz.timeline(filtered, show=True)

        App().mainloop()
        return 0


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    sys.exit(LogApp().run_cli())
