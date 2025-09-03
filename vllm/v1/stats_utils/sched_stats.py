import csv
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import vllm.envs as envs


@dataclass
class StepRecord:
    # Identifiers / timing
    step_id: int
    engine_time_start_ns: int = 0
    engine_time_end_ns: int = 0
    sched_dur_ms: float = 0.0
    exec_dur_ms: float = 0.0
    output_dur_ms: float = 0.0
    lat_between_steps_ms: float = 0.0

    # Scheduling state
    total_scheduled_tokens: int = 0

    num_newly_scheduled: int = 0
    newly_scheduled_request_ids: List[Any] = field(default_factory=list)

    num_running_requests: int = 0
    running_request_ids: List[Any] = field(default_factory=list)

    num_preempted_requests: int = 0
    total_preempted_tokens: int = 0
    preempted_request_ids: List[Any] = field(default_factory=list)
    preempted_tokens_by_request: Dict[str, int] = field(default_factory=dict)

    num_resumed_requests: int = 0
    total_resumed_tokens: int = 0
    resumed_request_ids: List[Any] = field(default_factory=list)
    resumed_tokens_by_request: Dict[str, int] = field(default_factory=dict)

    # KV cache / resources
    kv_usage_before_sched: Optional[float] = None
    kv_usage_after_sched: Optional[float] = None
    kv_usage_after_process: Optional[float] = None

    # Output / throughput
    est_avg_output_length: Optional[int] = None

    # Dynamic extra fields added via add_metric
    extras: Dict[str, Any] = field(default_factory=dict)


class SchedStatsCollector:
    _instance: Optional["SchedStatsCollector"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self, enable: bool, csv_path: Optional[str] = None, flush_interval: int = 0) -> None:
        self._enabled: bool = bool(enable)
        self._flush_interval: int = max(0, int(flush_interval or 0))
        self._lock = threading.Lock()
        self._records: Dict[int, StepRecord] = {}
        self._written_header: bool = False
        self._last_step_end_ns: Optional[int] = None

        self._csv_path = csv_path
        # Ensure directory exists if enabled
        if self._enabled:
            assert self._csv_path is not None, "csv_path is required"
            print(f"SchedStatsCollector: enabled, csv_path: {self._csv_path}")
            Path(self._csv_path).parent.mkdir(parents=True, exist_ok=True)

    # ----- Singleton accessors -----
    @staticmethod
    def get() -> "SchedStatsCollector":
        inst = SchedStatsCollector._instance
        if inst is None:
            with SchedStatsCollector._instance_lock:
                inst = SchedStatsCollector._instance
                if inst is None:
                    if envs.VLLM_V1_SCHED_STATS:
                        inst = SchedStatsCollector(
                            enable=True, csv_path=envs.VLLM_V1_SCHED_STATS_PATH, 
                            flush_interval=envs.VLLM_V1_SCHED_STATS_FLUSH_EVERY_N
                        )
                    else:
                        inst = _NoOpSchedStatsCollector()
                    SchedStatsCollector._instance = inst
        return inst

    @staticmethod
    def get_instance() -> "SchedStatsCollector":
        # Back-compat alias for get()
        return SchedStatsCollector.get()

    def is_enabled(self) -> bool:
        return self._enabled

    # ----- Per-step lifecycle -----
    def start_step(self, step_id: int, t_ns: Optional[int] = None, **initial_fields: Any) -> None:
        if not self._enabled:
            return
        start_ns = t_ns if t_ns is not None else time.perf_counter_ns()
        with self._lock:
            record = self._records.get(step_id)
            if record is None:
                record = StepRecord(step_id=step_id)
                self._records[step_id] = record

            record.engine_time_start_ns = start_ns
            # Compute latency between steps if previous step end is known
            if self._last_step_end_ns is not None:
                record.lat_between_steps_ms = max(
                    0.0, (start_ns - self._last_step_end_ns) / 1_000_000.0
                )
            # Apply any provided initial fields to known attributes or extras
            self._apply_initial_fields(record, initial_fields)

    def add_metric(self, step_id: int, key: str, value: Any) -> None:
        if not self._enabled:
            return
        with self._lock:
            record = self._records.get(step_id)
            if record is None:
                record = StepRecord(step_id=step_id)
                self._records[step_id] = record
            self._apply_metric(record, key, value)

    def add_metrics(self, step_id: int, metrics: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        with self._lock:
            record = self._records.get(step_id)
            if record is None:
                record = StepRecord(step_id=step_id)
                self._records[step_id] = record
            for key, value in metrics.items():
                self._apply_metric(record, key, value)

    def record_newly_scheduled(self, step_id: int, request_ids: List[Any]) -> None:
        if not self._enabled:
            return
        with self._lock:
            record = self._get_or_create(step_id)
            record.newly_scheduled_request_ids = list(request_ids)
            record.num_newly_scheduled = len(record.newly_scheduled_request_ids)

    def record_running(self, step_id: int, request_ids: List[Any]) -> None:
        if not self._enabled:
            return
        with self._lock:
            record = self._get_or_create(step_id)
            record.running_request_ids = list(request_ids)
            record.num_running_requests = len(record.running_request_ids)

    def record_preempted(self, step_id: int, items: List[Dict[str, Any]]) -> None:
        if not self._enabled:
            return
        with self._lock:
            record = self._get_or_create(step_id)
            for item in items or []:
                req_id = item.get("request_id")
                if req_id is None:
                    continue
                if req_id not in record.preempted_request_ids:
                    record.preempted_request_ids.append(req_id)
                tokens = int(item.get("preempted_tokens", 0) or 0)
                record.preempted_tokens_by_request[str(req_id)] = tokens
            record.num_preempted_requests = len(record.preempted_request_ids)

    def record_resumed(self, step_id: int, items: List[Dict[str, Any]]) -> None:
        if not self._enabled:
            return
        with self._lock:
            record = self._get_or_create(step_id)
            for item in items or []:
                req_id = item.get("request_id")
                if req_id is None:
                    continue
                if req_id not in record.resumed_request_ids:
                    record.resumed_request_ids.append(req_id)
                tokens = int(item.get("resumed_tokens", 0) or 0)
                record.resumed_tokens_by_request[str(req_id)] = tokens
            record.num_resumed_requests = len(record.resumed_request_ids)

    def end_step(self, step_id: int, t_ns: Optional[int] = None) -> None:
        if not self._enabled:
            return
        end_ns = t_ns if t_ns is not None else time.perf_counter_ns()
        with self._lock:
            record = self._records.get(step_id)
            if record is None:
                record = StepRecord(step_id=step_id)
                self._records[step_id] = record
            record.engine_time_end_ns = end_ns
            self._last_step_end_ns = end_ns

            # Periodic flush if configured
            if self._flush_interval > 0 and len(self._records) >= self._flush_interval:
                self._flush_locked()

    # ----- Output / lifecycle -----
    def flush(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._flush_locked()

    def finalize(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._flush_locked()

    # ----- Internal helpers -----
    def _get_or_create(self, step_id: int) -> StepRecord:
        record = self._records.get(step_id)
        if record is None:
            record = StepRecord(step_id=step_id)
            self._records[step_id] = record
        return record

    def _apply_initial_fields(self, record: StepRecord, fields: Dict[str, Any]) -> None:
        for key, value in (fields or {}).items():
            self._apply_metric(record, key, value)

    def _apply_metric(self, record: StepRecord, key: str, value: Any) -> None:
        # Known direct attributes
        if hasattr(record, key):
            setattr(record, key, value)
        else:
            record.extras[key] = value

    @staticmethod
    def _header_fields() -> List[str]:
        return [
            # Identity / timing
            "step_id",
            "sched_dur_ms",
            "exec_dur_ms",
            "output_dur_ms",
            "per_step_dur_ms",
            "lat_between_steps_ms",
            # Scheduling state
            "total_scheduled_tokens",
            "total_scheduled_reqs",
            "num_newly_scheduled",
            "newly_scheduled_request_ids",
            "num_running_requests",
            "running_request_ids",
            "num_preempted_requests",
            "total_preempted_tokens",
            "preempted_request_ids",
            "preempted_tokens_by_request",
            "num_resumed_requests",
            "total_resumed_tokens",
            "resumed_request_ids",
            "resumed_tokens_by_request",
            # KV / resources
            "kv_usage_before_sched",
            "kv_usage_after_sched",
            "kv_usage_after_process",
            # Output / estimates
            "est_avg_output_length",
            # Dynamic extras
            "extras",
        ]

    def _flush_locked(self) -> None:
        if not self._records:
            return
        header = self._header_fields()
        path = Path(self._csv_path)

        write_header = not self._written_header or not path.exists()
        # Sort by step id for deterministic output
        rows = [self._record_to_row(self._records[k]) for k in sorted(self._records.keys())]

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
                self._written_header = True
            for row in rows:
                writer.writerow(row)

        # Clear buffer after flushing
        self._records.clear()

    def _record_to_row(self, record: StepRecord) -> Dict[str, Any]:
        def dump_json(value: Any) -> str:
            try:
                return json.dumps(value, separators=(",", ":"))
            except Exception:
                return json.dumps(str(value))

        row: Dict[str, Any] = {
            "step_id": record.step_id,
            "sched_dur_ms": record.sched_dur_ms,
            "exec_dur_ms": record.exec_dur_ms,
            "output_dur_ms": record.output_dur_ms,
            "per_step_dur_ms": (record.engine_time_end_ns - record.engine_time_start_ns) / 1_000_000.0,
            "lat_between_steps_ms": record.lat_between_steps_ms,
            "total_scheduled_tokens": record.total_scheduled_tokens,
            "total_scheduled_reqs": record.num_newly_scheduled + record.num_running_requests + record.num_resumed_requests,
            "num_newly_scheduled": record.num_newly_scheduled,
            "newly_scheduled_request_ids": record.newly_scheduled_request_ids,
            "num_running_requests": record.num_running_requests,
            "running_request_ids": record.running_request_ids,
            "num_preempted_requests": record.num_preempted_requests,
            "total_preempted_tokens": record.total_preempted_tokens,
            "preempted_request_ids": record.preempted_request_ids,
            "preempted_tokens_by_request": dump_json(record.preempted_tokens_by_request),
            "num_resumed_requests": record.num_resumed_requests,
            "total_resumed_tokens": record.total_resumed_tokens,
            "resumed_request_ids": record.resumed_request_ids,
            "resumed_tokens_by_request": record.resumed_tokens_by_request,
            "kv_usage_before_sched": record.kv_usage_before_sched,
            "kv_usage_after_sched": record.kv_usage_after_sched,
            "kv_usage_after_process": record.kv_usage_after_process,
            "est_avg_output_length": record.est_avg_output_length,
            "extras": dump_json(record.extras) if record.extras else "{}",
        }
        return row


class _NoOpSchedStatsCollector(SchedStatsCollector):
    def __init__(self) -> None:
        super().__init__(enable=False)

    def is_enabled(self) -> bool:
        return False

    def start_step(self, step_id: int, t_ns: Optional[int] = None, **initial_fields: Any) -> None:
        return None

    def add_metric(self, step_id: int, key: str, value: Any) -> None:
        return None

    def add_metrics(self, step_id: int, metrics: Dict[str, Any]) -> None:
        return None

    def record_newly_scheduled(self, step_id: int, request_ids: List[Any]) -> None:
        return None

    def record_running(self, step_id: int, request_ids: List[Any]) -> None:
        return None

    def record_preempted(self, step_id: int, items: List[Dict[str, Any]]) -> None:
        return None

    def record_resumed(self, step_id: int, items: List[Dict[str, Any]]) -> None:
        return None

    def end_step(self, step_id: int, t_ns: Optional[int] = None) -> None:
        return None

    def flush(self) -> None:
        return None

    def finalize(self) -> None:
        return None
